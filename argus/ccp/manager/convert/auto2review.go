package convert

import (
	"bytes"
	"compress/gzip"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"

	xlog "github.com/qiniu/xlog.v1"
	BUCKET "qiniu.com/argus/argus/com/bucket"
	"qiniu.com/argus/censor/biz"
	"qiniu.com/argus/utility/censor"
	"qiniupkg.com/api.v7/kodo"

	"qiniu.com/argus/ccp/manager/client"
	"qiniu.com/argus/ccp/manager/proto"
)

const (
	SUG_PASS     = "PASS"
	SUG_REVIEW   = "REVIEW"
	SUG_BLOCK    = "BLOCK"
	SUG_DISABLED = "DISABLED"
)

const (
	CUR_VERSION = "2.0"
)

type ItemResult struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
	Faces []struct {
		BoundingBox struct {
			Pts   [][2]int `json:"pts"`
			Score float32  `json:"score"`
		} `json:"bounding_box"`
		Faces []struct {
			ID    string  `json:"id,omitempty"`
			Name  string  `json:"name,omitempty"`
			Score float32 `json:"score"`
			Group string  `json:"group,omitempty"`
		} `json:"faces,omitempty"`
	} `json:"faces"`
}

func (item *ItemResult) GetScore() float32 {

	if item.Score > 0 { // 有分返回分数
		return item.Score
	}

	sc := float32(0) // 没分的，取涉政最高分
	for _, f := range item.Faces {
		for _, ff := range f.Faces {
			if ff.Score > sc {
				sc = ff.Score
			}
		}
	}

	if sc > 0 { // 涉政最高分
		return sc
	}

	if item.Label == string(biz.REVIEW) ||
		item.Label == biz.POLITICIAN_FACE { // 疑似
		return float32(0.333333)
	}

	return float32(0.999999) // 保底
}

func (item *ItemResult) GetLabelInfos() []client.LabelInfo {

	labelInfos := make([]client.LabelInfo, 0)
	for _, faces := range item.Faces {
		pts := faces.BoundingBox.Pts
		for _, face := range faces.Faces {
			labelInfo := client.LabelInfo{
				Label: face.Name,
				Score: face.Score,
				Group: face.Group,
				Pts:   pts,
			}
			labelInfos = append(labelInfos, labelInfo)
		}
	}

	if len(labelInfos) <= 0 {
		labelInfo := client.LabelInfo{
			Label: item.Label,
			Score: item.GetScore(),
		}
		labelInfos = append(labelInfos, labelInfo)
	}

	return labelInfos
}

//================================================================//

type AutoResult struct {
	Code     int             `json:"code"`
	Mimetype string          `json:"mimetype"`
	Error    string          `json:"error,omitempty"`
	Result   json.RawMessage `json:"result,omitempty"` // biz.CensorResponse
}

type PfopVideoResult struct {
	Disable bool               `json:"disable"`
	Result  biz.CensorResponse `json:"result"`
}

type PfopImageResult struct {
	Disable bool               `json:"disable"`
	Result  biz.CensorResponse `json:"result"`
}

type PfopImageResultOld struct {
	Disable bool                   `json:"disable"`
	Result  censor.ImageCensorResp `json:"result"`
}

func ConvPfopOld2Review(rule *proto.Rule, uri string, pfopResult *PfopImageResultOld) *client.Entry {

	osmap := make(map[string]*client.OriginalSuggestionResult)
	for _, detail := range pfopResult.Result.Result.Details {

		oriSug := client.OriginalSuggestionResult{}
		oriSug.Labels = make([]client.LabelInfo, 0)

		if detail.Type == "pulp" {
			labelInfo := client.LabelInfo{}
			if detail.Label == 2 {
				labelInfo.Label = biz.PULP_NORMAL
				oriSug.Suggestion = SUG_PASS // 正常
			} else if detail.Label == 0 {
				labelInfo.Label = biz.PULP_PULP
				if detail.Review {
					oriSug.Suggestion = SUG_REVIEW
				} else {
					oriSug.Suggestion = SUG_BLOCK // 确定-色情-违规
				}
			} else {
				labelInfo.Label = biz.PULP_SEXY
				oriSug.Suggestion = SUG_REVIEW
			}

			// 保留原始标签/分类
			if detail.Class != "" {
				labelInfo.Label = detail.Class
			}
			labelInfo.Score = detail.Score
			oriSug.Labels = append(oriSug.Labels, labelInfo)
		} else if detail.Type == "terror" {
			labelInfo := client.LabelInfo{}
			if detail.Label == 0 {
				labelInfo.Label = biz.TERROR_NORMAL
				oriSug.Suggestion = SUG_PASS // 正常
			} else if detail.Label == 1 {
				labelInfo.Label = biz.TERROR_TERROR
				if detail.Review {
					oriSug.Suggestion = SUG_REVIEW
				} else {
					oriSug.Suggestion = SUG_BLOCK // 确定-违规
				}
			}

			// 保留原始标签/分类
			if detail.Class != "" {
				labelInfo.Label = detail.Class
			}
			labelInfo.Score = detail.Score
			oriSug.Labels = append(oriSug.Labels, labelInfo)
		} else if detail.Type == "politician" {

			faceDetails := make([]censor.FaceSearchDetail, 0)
			_ = convByJson(detail.More, &faceDetails)

			blockGroup := false
			for _, face := range faceDetails {

				if !blockGroup &&
					face.Value.Group != "" &&
					face.Value.Group != "domestic_statesman" &&
					face.Value.Group != "foreign_statesman" {
					blockGroup = true
				}

				labelInfo := client.LabelInfo{
					Label: face.Value.Name,
					Score: face.Value.Score,
					Group: face.Value.Group,
					Pts:   face.BoundingBox.Pts,
				}
				oriSug.Labels = append(oriSug.Labels, labelInfo)
			}

			labelInfo := client.LabelInfo{}
			if detail.Label == 0 {
				labelInfo.Label = biz.POLITICIAN_NORMAL
				oriSug.Suggestion = SUG_PASS // 正常
			} else if detail.Label == 1 {
				labelInfo.Label = biz.POLITICIAN_POLITICIAN
				if detail.Review || !blockGroup { // 涉政分组非禁用分组，归为疑似
					oriSug.Suggestion = SUG_REVIEW
				} else {
					oriSug.Suggestion = SUG_BLOCK // 确定-违规
				}
			}

			// 保留原始标签/分类
			if detail.Class != "" {
				labelInfo.Label = detail.Class
			}
			labelInfo.Score = detail.Score
			if len(oriSug.Labels) <= 0 {
				oriSug.Labels = append(oriSug.Labels, labelInfo)
			}
		}

		osmap[detail.Type] = &oriSug
	}

	ori := client.OriginalSuggestion{
		Source: "AUTOMATIC",
		Scenes: osmap,
	}

	if pfopResult.Disable {
		ori.Suggestion = SUG_DISABLED
	} else {
		ori.Suggestion = func() string {
			lvlArr := []string{ // 倒序，按优先级
				// SUG_DISABLED, // 场景里无此类别
				SUG_BLOCK,
				SUG_REVIEW,
				SUG_PASS,
			}
			for _, lvl := range lvlArr {
				for _, orisug := range ori.Scenes {
					if orisug.Suggestion == lvl {
						return lvl
					}
				}
			}
			return SUG_PASS
		}()
	}

	entry := client.Entry{
		SetID:    rule.RuleID,
		URIGet:   uri,
		MimeType: "IMAGE",
		Original: &ori,
		Version:  CUR_VERSION,
	}

	return &entry
}

func ConvPfop2Review(rule *proto.Rule, uri string, pfopResult *PfopImageResult) *client.Entry {
	entry := client.Entry{
		SetID:    rule.RuleID,
		URIGet:   uri,
		MimeType: strings.ToUpper(client.MT_IMAGE),
		Version:  CUR_VERSION,
	}

	osmap := make(map[string]*client.OriginalSuggestionResult)

	for sceneK, sceneV := range pfopResult.Result.Scenes {
		oriSug := client.OriginalSuggestionResult{}
		oriSug.Labels = make([]client.LabelInfo, 0)

		imgResp := biz.ImageSceneResponse{}
		_ = convByJson(sceneV, &imgResp)

		oriSug.Suggestion = strings.ToUpper(string(imgResp.Suggestion))

		respRes := ItemResult{}
		_ = convByJson(imgResp.Result, &respRes)

		oriSug.Labels = respRes.GetLabelInfos()
		osmap[string(sceneK)] = &oriSug
	}

	ori := client.OriginalSuggestion{
		Source:     "AUTOMATIC",
		Scenes:     osmap,
		Suggestion: strings.ToUpper(string(pfopResult.Result.Suggestion)),
	}

	if pfopResult.Disable {
		ori.Suggestion = SUG_DISABLED
	}
	entry.Original = &ori

	return &entry
}

func ConvPfopVideo2Review(rule *proto.Rule, uri string, pfopResult *PfopVideoResult) *client.Entry {
	entry := client.Entry{
		SetID:    rule.RuleID,
		URIGet:   uri,
		MimeType: strings.ToUpper(client.MT_VIDEO),
		Version:  CUR_VERSION,
	}

	osmap := make(map[string]*client.OriginalSuggestionResult)

	videoCutMap := make(map[int64]client.VideoCut)
	for sceneK, sceneV := range pfopResult.Result.Scenes {
		oriSug := client.OriginalSuggestionResult{}
		oriSug.Labels = make([]client.LabelInfo, 0)

		vidResp := biz.VideoSceneResult{}
		_ = convByJson(sceneV, &vidResp)

		oriSug.Suggestion = strings.ToUpper(string(vidResp.Suggestion))

		// 合并Cuts
		fillVideoCuts(context.Background(), videoCutMap,
			string(sceneK), vidResp.Segments)
		osmap[string(sceneK)] = &oriSug
	}

	// 取出CutArr
	entry.VideoCuts = getSortedCuts(videoCutMap)

	ori := client.OriginalSuggestion{
		Source:     "AUTOMATIC",
		Scenes:     osmap,
		Suggestion: strings.ToUpper(string(pfopResult.Result.Suggestion)),
	}

	if pfopResult.Disable {
		ori.Suggestion = SUG_DISABLED
	}

	entry.Original = &ori
	return &entry
}

func convBjob2ReviewItem(ctx context.Context, rule *proto.Rule, uri string, result *AutoResult) *client.Entry {
	xl := xlog.FromContextSafe(ctx)
	entry := client.Entry{
		SetID:    rule.RuleID,
		URIGet:   uri,
		MimeType: strings.ToUpper(result.Mimetype),
		Version:  CUR_VERSION,
	}

	if result.Code/100 == 2 {

		censorResp := biz.CensorResponse{}
		err := json.Unmarshal(result.Result, &censorResp)
		if err != nil {
			xl.Errorf("Unmarshal CensorResponse err, %+v", err)
			return &entry
		}

		osmap := make(map[string]*client.OriginalSuggestionResult)
		// 按MimeType解析结果
		if result.Mimetype == client.MT_IMAGE {
			for sceneK, sceneV := range censorResp.Scenes {
				oriSug := client.OriginalSuggestionResult{}
				oriSug.Labels = make([]client.LabelInfo, 0)
				imgResp := biz.ImageSceneResponse{}
				_ = convByJson(sceneV, &imgResp)

				oriSug.Suggestion = strings.ToUpper(string(imgResp.Suggestion))

				respRes := ItemResult{}
				_ = convByJson(imgResp.Result, &respRes)

				oriSug.Labels = respRes.GetLabelInfos()
				osmap[string(sceneK)] = &oriSug
			}
		} else if result.Mimetype == client.MT_VIDEO {
			videoCutMap := make(map[int64]client.VideoCut)
			for sceneK, sceneV := range censorResp.Scenes {
				oriSug := client.OriginalSuggestionResult{}
				oriSug.Labels = make([]client.LabelInfo, 0)
				vidResp := biz.VideoSceneResult{}
				err := convByJson(sceneV, &vidResp)
				if err != nil {
					xl.Errorf("VideoSceneResult err, %+v", err)
				}
				oriSug.Suggestion = strings.ToUpper(string(vidResp.Suggestion))

				// 合并Cuts
				fillVideoCuts(ctx, videoCutMap, string(sceneK), vidResp.Segments)
				osmap[string(sceneK)] = &oriSug
			}

			// 取出CutArr
			entry.VideoCuts = getSortedCuts(videoCutMap)
		}

		ori := client.OriginalSuggestion{
			Source:     "AUTOMATIC",
			Scenes:     osmap,
			Suggestion: strings.ToUpper(string(censorResp.Suggestion)),
		}

		entry.Original = &ori
	}

	return &entry
}

func fillVideoCuts(ctx context.Context, videoCutMap map[int64]client.VideoCut,
	scene string, segments []biz.SegmentResult) {

	for _, seg := range segments {
		for _, cut := range seg.Cuts {

			cutRes := ItemResult{}
			_ = convByJson(cut.Result, &cutRes)

			vc, ok := videoCutMap[cut.Offset]
			if ok {
				// 有则合并
				if vc.Original == nil || vc.Original.Scenes == nil {
					oriSugRes := make(map[string]*client.OriginalSuggestionResult)
					vc.Original = &client.OriginalSuggestion{
						Source: "AUTOMATIC",
						Scenes: oriSugRes,
					}
				}
				vc.Original.Scenes[scene] = &client.OriginalSuggestionResult{
					Suggestion: strings.ToUpper(string(cut.Suggestion)),
					Labels:     cutRes.GetLabelInfos(),
				}
			} else {
				// 无则新增
				oriSugRes := make(map[string]*client.OriginalSuggestionResult)
				oriSugRes[scene] = &client.OriginalSuggestionResult{
					Suggestion: strings.ToUpper(string(cut.Suggestion)),
					Labels:     cutRes.GetLabelInfos(),
				}
				videoCutMap[cut.Offset] = client.VideoCut{
					Uri:    cut.URI,
					Offset: cut.Offset,
					Original: &client.OriginalSuggestion{
						Source:     "AUTOMATIC",
						Scenes:     oriSugRes,
						Suggestion: strings.ToUpper(string(cut.Suggestion)),
					},
				}
			}

			// 合并Cut结果
			vc, ok = videoCutMap[cut.Offset]
			if ok &&
				(cut.Suggestion == biz.REVIEW && vc.Original.Suggestion == strings.ToUpper(string(biz.PASS))) ||
				(cut.Suggestion == biz.BLOCK && (vc.Original.Suggestion == strings.ToUpper(string(biz.REVIEW)) ||
					vc.Original.Suggestion == strings.ToUpper(string(biz.PASS)))) {
				vc.Original.Suggestion = strings.ToUpper(string(cut.Suggestion))
			}
		}
	}
}

func getSortedCuts(videoCutMap map[int64]client.VideoCut) []client.VideoCut {

	offsets := Offsets{
		offsets: make([]int64, 0, len(videoCutMap)),
	}
	for k := range videoCutMap {
		offsets.offsets = append(offsets.offsets, k)
	}
	sort.Sort(offsets)

	vcuts := make([]client.VideoCut, 0, len(videoCutMap))
	for _, offset := range offsets.offsets {
		vcuts = append(vcuts, videoCutMap[offset])
	}

	return vcuts
}

func ConvBjob2Review(ctx context.Context, rule *proto.Rule, kodo *kodo.Config, ak, sk, bucket, domain string, fileKeys []string) []string {
	xl := xlog.FromContextSafe(ctx)
	outputFileKeys := make([]string, 0, len(fileKeys))

	for _, fileKey := range fileKeys {
		xl.Infof("handle bjob result file key: %s", fileKey)
		handler := Handler{
			Config: *kodo,
		}
		iter := handler.ScannerBjobResult(ctx,
			ak, sk, bucket, domain, []string{fileKey},
		)

		buf := bytes.NewBuffer(make([]byte, 0, 1024*1024*256))
		for {
			linestr, ok := iter.Next(ctx)

			xl.Debugf("linestr, %s, %v", linestr, ok)

			if !ok {
				break
			}
			if linestr == "" {
				continue
			}
			strs := strings.Split(linestr, "\t")
			if len(strs) < 2 {
				continue
			}
			resultInside := AutoResult{}
			err := json.Unmarshal([]byte(strs[1]), &resultInside)
			if err != nil {
				xl.Warnf("linestr parse json failed, %s, %+v", linestr, err)
			}
			entry := convBjob2ReviewItem(ctx, rule, strs[0], &resultInside)
			jsonStr, err := json.Marshal(entry)
			if err != nil {
				xl.Warnf("Marshal json failed, %s, %v", linestr, ok)
				continue
			}
			buf.WriteString(strs[0])
			buf.WriteString("\t")
			buf.Write(jsonStr)
			buf.WriteString("\n")
		}
		gzBuf := gz(buf)
		outputFileName := fmt.Sprintf("%s_%s", time.Now().Format("20060102150405"), fileKey)
		xl.Infof("outputFileName: %s", outputFileName)
		outputFileKey, err := BUCKET.Bucket{
			Config: BUCKET.Config{
				Config: *kodo,
				Bucket: bucket,
				Domain: domain,
			}.New(ak, sk, 0, bucket, rule.RuleID),
		}.Save(ctx, outputFileName, gzBuf, int64(gzBuf.Len()))
		if err != nil {
			xl.Errorf("Save failed: %s", outputFileKey)
			continue
		}
		outputFileKeys = append(outputFileKeys, outputFileKey)
	}

	return outputFileKeys
}

func gz(buf *bytes.Buffer) *bytes.Buffer {
	buf2 := bytes.NewBuffer(nil)
	w := gzip.NewWriter(buf2)
	defer w.Close()
	_, _ = w.Write(buf.Bytes())
	w.Flush()
	return buf2
}

func convByJson(src interface{}, dest interface{}) error {

	tmpbs, err := json.Marshal(src)
	if err != nil {
		return err
	}

	return json.Unmarshal(tmpbs, dest)
}

//====offsets

type Offsets struct {
	offsets []int64
}

func (ofs Offsets) Len() int {
	return len(ofs.offsets)
}

func (ofs Offsets) Less(i, j int) bool {
	if ofs.offsets[i] < ofs.offsets[j] {
		return true
	}
	return false
}

func (ofs Offsets) Swap(i, j int) {
	ofs.offsets[i], ofs.offsets[j] = ofs.offsets[j], ofs.offsets[i]
}

package service

import (
	"archive/tar"
	"container/list"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/pkg/errors"
	httputil "github.com/qiniu/http/httputil.v1"
	xlog "github.com/qiniu/xlog.v1"
)

var (
	ErrListEmpty = errors.New("List empty")
)

const (
	IMAGE_HPTP_ONLY = "hptp_only"
	IMAGE_QJTP_ONLY = "qjtp_only"
	IMAGE_HPTP_QJTP = "hptp_qjtp"
)

const (
	resourcePrefix            = "resource"
	ftpPrefix                 = "ftp"
	tmpPrefix                 = "tmp"
	videoPrefix               = "video"
	defaultRate               = 1
	defaultWorkThreads        = 1
	defailtFrontOffset        = 5.0
	defaultVideoCacheDuration = 30 * 24 * 3600 // 7 days
	defailtBackOffset         = 5.0
	topNCapture               = 6

	defaultZhatucheSearchDuration = 120 // 查找前120s， 后120s
)

type CameraConfig struct {
	CameraID   string               `json:"camera_id"`
	CameraIP   string               `json:"camera_ip"`
	CameraInfo string               `json:"camera_info"`
	GPS        [2]float32           `json:"gps"`
	LanePTS    map[string][4][2]int `json:"lane_pts"`
	DvrConfig  DvrConfig            `json:"dvr_config"`
	ImageType  string               `json:"image_type"`
}

type MgrConfig struct {
	EvalConfig      `json:"default"`
	Evals           map[string]EvalConfig `json:"evals"`
	Cameras         []CameraConfig        `json:"cameras"`
	WorkerThreads   int                   `json:"worker_threads"`
	FileServer      string                `json:"file_server"`
	SmbServer       string                `json:"smb_server"`
	Workspace       string                `json:"workspace"`
	VideoCache      int                   `json:"video_cache"`
	Delay           int                   `json:"delay"`
	FetchInterval   int                   `json:"fetch_interval"`   // by second
	ArchiveInterval int                   `json:"archive_interval"` // by second
	CaptureInterval int                   `json:"capture_interval"` // by second
}

func (c MgrConfig) Get(cmd string) EvalConfig {
	if cc, ok := c.Evals[cmd]; ok {
		if cc.Host == "" {
			cc.Host = c.EvalConfig.Host
		}
		if cc.Timeout == 0 {
			cc.Timeout = c.EvalConfig.Timeout
		}
		cc.Timeout = c.Timeout
		return cc
	}
	return c.EvalConfig
}

type Video struct {
	Path     string
	DeviceIP string
	Start    time.Time
	End      time.Time
}

type Manager struct {
	MgrConfig

	Zhatu
	Zhongxing
	Jiaoguan
	LastFetch time.Time
	Videos    []*Video
	List      *list.List
	ListLock  sync.Mutex
	Lock      sync.Mutex
}

func NewManager(cfg MgrConfig) (*Manager, error) {
	mgr := &Manager{
		MgrConfig: cfg,
		List:      list.New(),
	}

	if mgr.VideoCache == 0 {
		mgr.VideoCache = defaultVideoCacheDuration
	}

	if mgr.WorkerThreads == 0 {
		mgr.WorkerThreads = defaultWorkThreads
	}

	const (
		_CmdZhatu     = "zhatu"
		_CmdZhongxing = "zhongxing"
		_CmdJiaoguan  = "jiaoguan"
	)
	{
		conf := cfg.Get(_CmdZhatu)
		mgr.Zhatu = NewZhatu(conf)
	}
	{
		conf := cfg.Get(_CmdZhongxing)
		mgr.Zhongxing = NewZhongxing(conf)
	}
	{
		conf := cfg.Get(_CmdJiaoguan)
		mgr.Jiaoguan = NewJiaoguan(conf)
	}
	if _, err := os.Stat(cfg.Workspace); err != nil && os.IsNotExist(err) {
		os.Mkdir(cfg.Workspace, 0755)
	}
	if _, err := os.Stat(path.Join(cfg.Workspace, resourcePrefix)); err != nil && os.IsNotExist(err) {
		os.Mkdir(path.Join(cfg.Workspace, resourcePrefix), 0755)
	}
	if err := mgr.InitVideo(); err != nil {
		return nil, err
	}
	go func() {
		ts := 0
		for {
			for _, camera := range mgr.Cameras {
				if camera.DvrConfig.DeviceIP != "" {
					go func(t int, ip string) {
						if t%1800 == 0 {
							mgr.clearVideoCache(context.Background(), ip)
						}
					}(ts, camera.DvrConfig.DeviceIP)
				}
			}
			go func(t int) {
				if t%cfg.FetchInterval == 0 {
					for _, camera := range mgr.Cameras {
						mgr.Fetch(context.Background(), time.Now().Add((-1)*time.Duration(cfg.Delay)*time.Second), camera.DvrConfig)
					}
				}
			}(ts)
			go func(t int) {
				if t%cfg.ArchiveInterval == 0 {
					mgr.Archive(context.Background(), time.Now().Add((-1)*time.Duration(cfg.ArchiveInterval*2)*time.Second).Format("20060102150405"), time.Now().Format("20060102150405"), false)
				}
			}(ts)
			if ts%cfg.CaptureInterval == 0 {
				for _, camera := range mgr.Cameras {
					go func(t int, camera CameraConfig) {
						ctx := context.Background()
						mgr.Capture(ctx, time.Now().Add((-1)*time.Duration(cfg.CaptureInterval*2)*time.Second).Format("20060102150405"), camera.CameraID, camera.CameraIP, cfg.CaptureInterval*2)
					}(ts, camera)
				}
			}
			ts++
			time.Sleep(time.Second)
		}
	}()

	for i := 0; i < mgr.WorkerThreads; i++ {
		go func() {
			for {
				ctx := context.Background()
				if err := mgr.Consume(ctx); err != nil {
					time.Sleep(time.Second)
				}
			}
		}()
	}
	return mgr, nil
}

func (m *Manager) cmd(ctx context.Context, args []string) error {
	xl := xlog.FromContextSafe(ctx)
	cmd := exec.Command(args[0], args[1:]...)
	if err := cmd.Start(); err != nil {
		return err
	}
	{
		done := make(chan bool)
		go func() {
			for {
				select {
				case <-done:
					goto _end
				case <-ctx.Done():
					cmd.Process.Kill()
				}
			}
		_end: // NOTHING
		}()
		defer func() {
			done <- true
		}()
	}
	if err := cmd.Wait(); err != nil {
		xl.Warnf("cmd finish. %v", err)
		err = fmt.Errorf("exec cmd (%v) failed, err: %s", args, err)
		return err
	}
	//xl.Infof("cmd [ %s ] finish.", strings.Join(args, " "))

	return nil
}

var xl = xlog.NewWith("main")

func (m *Manager) InitVideo() (err error) {
	err = filepath.Walk(path.Join(m.Workspace, videoPrefix), func(p string, info os.FileInfo, err error) error {
		if info != nil && !info.IsDir() {
			video := Video{Path: p}
			strs := strings.Split(strings.TrimSuffix(info.Name(), path.Ext(video.Path)), "_")
			if video.Start, err = time.Parse("20060102150405", strs[0]); err != nil {
				return err
			}
			if video.End, err = time.Parse("20060102150405", strs[1]); err != nil {
				return err
			}
			video.DeviceIP = path.Base(path.Dir(p))

			m.Lock.Lock()
			xl.Debug("Video:", video)
			m.Videos = append(m.Videos, &video)
			m.Lock.Unlock()
		}
		return nil
	})
	return
}

func (m *Manager) clearVideoCache(ctx context.Context, deviceIP string) (err error) {

	timestamp := time.Now().Add((-1) * time.Duration(m.VideoCache) * time.Second).Format("2006010215")

	resource := path.Join(m.Workspace, videoPrefix, deviceIP, timestamp[:10])
	err = filepath.Walk(resource, func(p string, info os.FileInfo, err error) error {
		if info != nil && !info.IsDir() {
			for index, video := range m.Videos {
				if video.Path == p {
					m.ListLock.Lock()
					m.Videos = append(m.Videos[:index], m.Videos[index+1:]...)
					m.ListLock.Unlock()
				}
			}
		}
		return nil
	})
	xlog.FromContextSafe(ctx).Infof("delete video folder %s", resource)
	os.RemoveAll(resource)
	return
}

func (m *Manager) Fetch(ctx context.Context, tm time.Time, dvrConfig DvrConfig) (vd *Video, err error) {
	// fetch very interval
	xl := xlog.FromContextSafe(ctx)

	v := m.Find(tm, dvrConfig.DeviceIP)
	if v != nil {
		return
	}
	timestamp := tm.Format("20060102150405")
	dir := path.Join(m.Workspace, tmpPrefix, timestamp)
	os.MkdirAll(dir, 0755)
	defer func(dir string) {
		os.RemoveAll(dir)
	}(dir)
	// fetch: dvr download
	args := []string{"./DvrDL", "-uid", "pudong", "-start_time", timestamp, "-prefix", dir + "/", "-device_ip", dvrConfig.DeviceIP,
		"-channel_index", strconv.Itoa(dvrConfig.ChannelIndex), "-user_id", dvrConfig.UserID, "-user_pwd", dvrConfig.UserPwd, "-port", strconv.Itoa(dvrConfig.Port)}

	if err = m.cmd(ctx, args); err != nil {
		xl.Errorf("Fetch failed, err: %s", err)
		return nil, err
	}

	var found bool
	var video Video
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			found = true
		}
		video.Path = info.Name()
		return nil
	})
	if err != nil {
		xl.Errorf("file path walk error: %s", err)
		return nil, err
	}
	if !found {
		return
	}

	folder := path.Join(m.Workspace, videoPrefix, dvrConfig.DeviceIP, tm.Format("2006010215"))
	if _, err := os.Stat(folder); err != nil && os.IsNotExist(err) {
		os.MkdirAll(folder, 0755)
	}
	os.Rename(path.Join(dir, video.Path), path.Join(folder, video.Path))
	strs := strings.Split(strings.Trim(video.Path, path.Ext(video.Path)), "_")
	video.Start, _ = time.Parse("20060102150405", strs[0])
	video.End, _ = time.Parse("20060102150405", strs[1])
	video.Path = path.Join(folder, video.Path)
	video.DeviceIP = dvrConfig.DeviceIP

	xl.Debug("Video:", video)
	m.Lock.Lock()
	m.Videos = append(m.Videos, &video)
	m.Lock.Unlock()

	return &video, nil
}

func (m *Manager) Find(tm time.Time, deviceIP string) *Video {
	m.Lock.Lock()
	defer m.Lock.Unlock()

	for _, video := range m.Videos {
		if video.DeviceIP == deviceIP && video.Start.Before(tm) && video.End.After(tm) {
			return video
		}
	}
	return nil
}

func (m *Manager) GenPic(ctx context.Context, video *Video, dir string, rate int, start, end time.Time, prefix string) (images []string, err error) {
	xl := xlog.FromContextSafe(ctx)

	startOffset := int(start.Sub(video.Start).Seconds())
	endtOffset := int(end.Sub(video.Start).Seconds())
	args := []string{
		"./ExtractFrames",
		"-i", video.Path,
		"-ss", strconv.Itoa(startOffset),
		"-to", strconv.Itoa(endtOffset),
		"-r", strconv.Itoa(rate),
		"-prefix", dir + "/" + prefix}
	xl.Debug("exec", args)
	err = m.cmd(ctx, args)
	if err != nil {
		xl.Errorf("fail to do cmd %s, err: %s", strings.Join(args, " "), err)
		return nil, err
	}

	err = filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if !info.IsDir() {
			name := info.Name()
			if strings.HasPrefix(name, prefix) {
				newName := fmt.Sprintf("%s-%d.jpg", prefix, len(images))
				err = os.Rename(filepath.Join(dir, name), filepath.Join(dir, newName))
				if err != nil {
					xl.Error("rename image", err)
				}
				images = append(images, path.Join(dir, newName))
			}
		}
		return nil
	})
	if err != nil {
		xl.Errorf("file path walk error: %s", err)
		return
	}

	return
}

type Image struct {
	URI string `json:"uri"`
	PTS [4]int `json:"pts"`
}

type DvrConfig struct {
	DeviceIP     string `json:"device_ip"`
	ChannelIndex int    `json:"channel_index"`
	UserID       string `json:"user_id"`
	UserPwd      string `json:"user_pwd"`
	Port         int    `json:"port"`
}

type Capture struct {
	ID          string  `json:"id,omitempty"`
	Time        string  `json:"time"`
	CameraID    string  `json:"camera_id"`
	CameraInfo  string  `json:"camera_info"`
	LicenceID   string  `json:"licence_id"`
	LicenceType string  `json:"licence_type"`
	Lane        int     `json:"lane"`
	Result      int     `json:"result"`
	Score       float32 `json:"score"`
	Coordinate  struct {
		GPS [2]float32 `json:"gps"`
	} `json:"coordinate"`
	LanePTS   map[string][4][2]int `json:"lane_pts"`
	DvrConfig DvrConfig            `json:"dvr_config"`
	Resource  struct {
		CaptureImages []Image  `json:"capture_images"`
		Images        []Image  `json:"images"`
		Videos        []string `json:"videos"`
	} `json:"resource"`
	ImageType string `json:"-"`
}

func (m *Manager) Produce(cap Capture) {
	m.ListLock.Lock()
	defer m.ListLock.Unlock()

	m.List.PushBack(cap)
	return
}

type checkVideoResult struct {
	video       *Video
	images      []string
	startOffset int
}

// 传入起始时间，结束时间，找出对应磁盘上面的视频文件和截图
func (m *Manager) checkVideo(ctx context.Context, start, end time.Time, dir string, rate int, dvrConfig DvrConfig) (r []checkVideoResult, err error) {
	var (
		videoStart, videoEnd *Video
	)
	xl := xlog.FromContextSafe(ctx)

	videoStart = m.Find(start, dvrConfig.DeviceIP)
	if videoStart == nil {
		if videoStart, err = m.Fetch(ctx, start, dvrConfig); err != nil || videoStart == nil {
			xl.Errorf("fetch video start %v failed, error: %v", start, err)
			return
		}
	}
	videoEnd = m.Find(end, dvrConfig.DeviceIP)
	if videoEnd == nil {
		if videoEnd, err = m.Fetch(ctx, end, dvrConfig); err != nil || videoEnd == nil {
			xl.Errorf("fetch video end %v failed, error: %s", end, err)
			return
		}
	}

	if videoEnd == videoStart {
		images, err := m.GenPic(ctx, videoStart, dir, rate, start, end, "video1")
		if err != nil {
			return nil, err
		}
		r = append(r, checkVideoResult{video: videoStart, images: images, startOffset: int(start.Sub(videoStart.Start).Seconds())})
		return r, nil
	}

	images, err := m.GenPic(ctx, videoStart, dir, rate, start, videoStart.End, "video1")
	if err != nil {
		return nil, err
	}
	r = append(r, checkVideoResult{video: videoStart, images: images, startOffset: int(start.Sub(videoStart.Start).Seconds())})

	images, e := m.GenPic(ctx, videoEnd, dir, rate, videoEnd.Start, end, "video2")
	if e != nil {
		err = e
		return
	}
	r = append(r, checkVideoResult{video: videoEnd, images: images, startOffset: 0})

	return
}

func (m *Manager) cutVideo(ctx context.Context, video *Video, start, end time.Time, filepath string) (err error) {
	xl := xlog.FromContextSafe(ctx)
	startOffset := int(start.Sub(video.Start).Seconds())
	endtOffset := int(end.Sub(video.Start).Seconds())
	args := []string{
		"./ExtractVideo",
		"-i", video.Path,
		"-ss", strconv.Itoa(startOffset),
		"-to", strconv.Itoa(endtOffset),
		"-o", filepath}
	xl.Debug("exec", args)
	err = m.cmd(ctx, args)
	if err != nil {
		xl.Errorf("fail to do cmd %s, err: %s", strings.Join(args, " "), err)
		return err
	}
	return
}

func (m *Manager) ConsumeIllegalCapture(ctx context.Context, cap *Capture, dir string, resource string) (err error) {
	xl := xlog.FromContextSafe(ctx)
	captureTime, _ := time.Parse("20060102150405", cap.Time)
	startTime := captureTime.Add(time.Duration(-defaultZhatucheSearchDuration) * time.Second)
	endTime := captureTime.Add(time.Duration(defaultZhatucheSearchDuration) * time.Second)
	xl.Debugf("checkVideo captureTime:%v startTime:%v endTime:%v", captureTime, startTime, endTime)
	videos, err := m.checkVideo(ctx, startTime, endTime, dir, defaultRate, cap.DvrConfig)
	if err != nil {
		xl.Error("failed to check video, err:", err)
		return err
	}
	xl.Debugf("checkVideo result:%#v", videos)
	var parts []*zhatuPart
	for _, video := range videos {
		p, err := m.searchVideo(ctx, captureTime, dir, defaultRate, video)
		if err != nil {
			xl.Error("failed to check video, err:", err)
			return err
		}
		xl.Debugf("searchVideo result:%#v %#v", video.video, p)
		parts = append(parts, p...)
	}
	sort.Slice(parts, func(i, j int) bool {
		return parts[i].distance < parts[j].distance
	})
	if len(parts) > 2 {
		parts = parts[:2]
	}
	for _, part := range parts {
		// 起始时间 = 12分钟视频起始时间+截取的2分钟视频片段本身的偏移值+片段偏移值
		startTime := part.video.video.Start.Add(time.Duration(part.startOffset+part.video.startOffset) * time.Second)
		endTime := part.video.video.Start.Add(time.Duration(part.endOffset+part.video.startOffset) * time.Second)
		if endTime.Sub(part.video.video.End) > 0 {
			endTime = part.video.video.End
		}
		newPath := path.Join(resource, part.partName)
		err = m.cutVideo(ctx, part.video.video, startTime, endTime, newPath)
		if err != nil {
			xl.Errorf("fail to cut video %s, err: %s", newPath, err)
		} else {
			cap.Resource.Videos = append(cap.Resource.Videos, m.FileServer+strings.TrimPrefix(newPath, m.Workspace))
		}
	}

	var images []zhatuPartImage
	for _, part := range parts {
		images = append(images, part.images...)
	}
	sort.Slice(images, func(i, j int) bool {
		return images[i].score > images[j].score
	})
	if len(images) > topNCapture {
		images = images[:topNCapture]
	}
	for _, image := range images {
		newPath := path.Join(resource, path.Base(image.uri))
		os.Rename(image.uri, newPath)

		cap.Resource.Images = append(cap.Resource.Images, Image{
			URI: m.FileServer + strings.TrimPrefix(newPath, m.Workspace),
			PTS: image.pts,
		})
		if image.score > cap.Score {
			cap.Score = image.score
		}
	}
	return nil
}

func (m *Manager) Consume(ctx context.Context) (err error) {
	xl := xlog.FromContextSafe(ctx)

	m.ListLock.Lock()
	front := m.List.Front()
	if front == nil {
		m.ListLock.Unlock()
		return ErrListEmpty
	}
	cap := m.List.Remove(front).(Capture)
	m.ListLock.Unlock()

	// start to check capture image
	dir := path.Join(m.Workspace, tmpPrefix, cap.ID)
	os.MkdirAll(dir, 0755)
	defer func(dir string) {
		os.RemoveAll(dir)
	}(dir)
	var (
		found   bool
		illegal bool
	)
	resource := path.Join(m.Workspace, resourcePrefix, cap.Time[:8], cap.Time[8:10], cap.ID)
	os.MkdirAll(resource, 0755)

	// 龙东华夏卡口图片分qjtp（全景图片）/hptp（高清图片）两种，但qjtp可能不存在
	// 需要对qjtp先尝试去下载

	if cap.ImageType != IMAGE_HPTP_ONLY {
		var images []Image
		for _, image := range cap.Resource.CaptureImages {
			image.URI = strings.Replace(image.URI, "hptp", "qjtp", 1)
			newPath := path.Join(dir, path.Base(image.URI))
			if err = download(m.SmbServer+image.URI, newPath); err != nil {
				xl.Warnf("failed to fild file: %s, err: %s", image.URI, err)
				continue
			}
			images = append(images, image)
		}
		switch cap.ImageType {
		case IMAGE_QJTP_ONLY:
			cap.Resource.CaptureImages = images
		case IMAGE_HPTP_QJTP:
			cap.Resource.CaptureImages = append(cap.Resource.CaptureImages, images...)
		}
	}

	for _, image := range cap.Resource.CaptureImages {
		newPath := path.Join(dir, path.Base(image.URI))
		if _, e := os.Stat(newPath); e != nil && os.IsNotExist(e) {
			if err = download(m.SmbServer+image.URI, newPath); err != nil {
				xl.Errorf("failed to download image: %s, err: %s", m.SmbServer+image.URI, err)
				return err
			}
		}

		var req EvalZhatuReq
		req.Data.URI = m.FileServer + strings.TrimPrefix(newPath, m.Workspace)
		req.Data.Attribute.Name = path.Base(req.Data.URI)
		var exist bool
		req.Data.Attribute.LanePTS, exist = cap.LanePTS[req.Data.Attribute.Name[14:16]]
		if !exist {
			// if lane is not exsit in the
			xl.Errorf("Zhatu eval lane %v not exist", req.Data.Attribute.Name[14:16])
			continue
		}
		resp, e := m.Zhatu.Eval(ctx, req)
		if e != nil {
			err = e
			xl.Error("Zhatu eval error:", err)
			return
		}
		for _, detect := range resp.Result.Detections {
			if detect.Label == ZhatuTypeCovered || detect.Label == ZhatuTypeUncovered {
				cap.Result = detect.Label
				if detect.Score > cap.Score {
					cap.Score = detect.Score
				}
				imgPath := path.Join(resource, path.Base(newPath))
				os.Rename(newPath, imgPath)
				cap.Resource.Images = append(cap.Resource.Images, Image{
					URI: m.FileServer + strings.TrimPrefix(imgPath, m.Workspace),
					PTS: detect.PTS,
				})
				found = true
				if detect.Label == ZhatuTypeUncovered {
					illegal = true
				}
			}
		}
	}
	if !CameraMatchFilters(cap.ID, cap.LicenceID) {
		if illegal {
			xl.Debugf("Ignore by cause 非沪牌: %#v", cap)
		}
		found = false
	}

	if !found {
		os.RemoveAll(resource)
		return
	}
	cap.Resource.CaptureImages = make([]Image, 0)
	xl.Debugf("Found capture: %#v, illegal: %v", cap, illegal)

	if illegal {
		m.ConsumeIllegalCapture(ctx, &cap, dir, resource)
	}

	xl.Debugf("Success!, cap: %#v", cap)
	if err = m.Zhongxing.Zhatu(ctx, cap); err != nil {
		xl.Error("Call zhonging err:", err)
		return
	}
	return
}

func (m *Manager) Capture(ctx context.Context, startTime, cameraID, cameraIP string, duration int) (err error) {
	xl := xlog.FromContextSafe(ctx)
	var (
		found  bool
		camera CameraConfig
	)
	xl.Debugf("Start to capture cameraID %s, from %s, duration %d seconds", cameraID, startTime, duration)
	for _, cm := range m.Cameras {
		if cm.CameraID == cameraID && cm.CameraIP == cameraIP {
			found = true
			camera = cm
			break
		}
	}
	if !found {
		return httputil.NewError(http.StatusBadRequest, "camera not found")
	}
	tm, err := time.Parse("20060102150405", startTime)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, "invalid start_time")
	}
	resp, err := m.Jiaoguan.Jiaoguan(ctx, JiaoguanReq{
		CameraID:  cameraID,
		CameraIP:  cameraIP,
		StartTime: tm.Unix() - 3600*8,
		Duration:  duration,
	})
	if err != nil {
		xl.Errorf("failed to fetch cameraID: %s, from %s, duration %d seconds", cameraID, startTime, duration)
		return
	}
	caps := make(map[string]Capture, 0)
	for _, img := range resp.Images {

		if img.LicenceType != licenceTypeYellow {
			continue
		}
		tt, _ := time.Parse("2006-01-02T15:04:05", img.TimestampStr)
		cap := Capture{
			ID:          img.ID,
			Time:        tt.Format("20060102150405"),
			CameraID:    img.CameraID,
			CameraInfo:  camera.CameraInfo,
			DvrConfig:   camera.DvrConfig,
			LanePTS:     camera.LanePTS,
			LicenceID:   img.LicenceID,
			LicenceType: img.LicenceType,
			Lane:        img.Lane,
			ImageType:   camera.ImageType,
		}
		if len(m.Cameras) > 0 {
			cap.Coordinate.GPS = m.Cameras[0].GPS
		}
		if len(img.FullImage) > 0 {
			cap.Resource.CaptureImages = append(cap.Resource.CaptureImages, Image{URI: path.Join(resp.BaseURL, img.FullImage)})
		}
		if len(img.LicenceImage) > 0 {
			cap.Resource.CaptureImages = append(cap.Resource.CaptureImages, Image{URI: path.Join(resp.BaseURL, img.LicenceImage)})
		}
		if len(cap.Resource.CaptureImages) > 0 {
			cc, exist := caps[cap.LicenceID]
			if exist && cc.Time == cap.Time && cc.CameraID == cap.CameraID && cc.ID != cap.ID {
				cap.Resource.CaptureImages = append(cap.Resource.CaptureImages, cc.Resource.CaptureImages...)
			}
			caps[cap.LicenceID] = cap
		}
	}
	for _, cap := range caps {
		xl.Debugf("Add capture: %#v", cap)
		m.Produce(cap)
	}
	return
}

func (m *Manager) Archive(ctx context.Context, startTime, endTime string, illegal bool) (err error) {
	xl := xlog.FromContextSafe(ctx)

	_, err = time.Parse("20060102150405", startTime)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, "invalid start_time")
	}
	_, err = time.Parse("20060102150405", endTime)
	if err != nil {
		return httputil.NewError(http.StatusBadRequest, "invalid end_time")
	}

	xl.Debugf("Start to archive from %s to %s", startTime, endTime)
	caps, err := m.Zhongxing.Report(ctx, startTime, endTime)
	if err != nil {
		xl.Error("Call zhonging.report err:", err)
		return httputil.NewError(http.StatusBadRequest, "failed to call zhongxin.report")
	}

	for _, c := range caps {
		if illegal && c.Result == 0 { // if only need illegal
			continue
		}
		// prepare tar file contents
		type _fileToArchive struct {
			Name string
			Body []byte
		}
		var filesToArchive = make([]_fileToArchive, 0)
		zhatucheJson, err := json.Marshal(c)
		if err != nil {
			xl.Error(err)
			return err
		}
		filesToArchive = append(filesToArchive, _fileToArchive{
			Name: "zhatuche.json",
			Body: zhatucheJson,
		})
		resourceDir := path.Join(m.Workspace, resourcePrefix, c.Time[:8], c.Time[8:10], c.ID)
		for _, image := range c.Resource.Images {
			data, err := ioutil.ReadFile(path.Join(resourceDir, image.FileName))
			if err != nil {
				xl.Error(err)
				return err
			}
			filesToArchive = append(filesToArchive, _fileToArchive{
				Name: image.FileName,
				Body: data,
			})
		}
		for vid, video := range c.Resource.Videos {
			data, err := ioutil.ReadFile(path.Join(resourceDir, path.Base(video)))
			if err != nil {
				xl.Error(err)
				return err
			}
			filesToArchive = append(filesToArchive, _fileToArchive{
				Name: fmt.Sprintf("%s_%s-%d.mp4", c.ID, c.LicenceID, vid+1),
				Body: data,
			})
		}

		// tar files
		tarName := fmt.Sprintf("%s-%.14s", c.LicenceID, c.Time)
		fullTarName := path.Join(m.Workspace, ftpPrefix, tarName)
		fullTarName = fullTarName + ".tar"
		tarFile, err := os.Create(fullTarName)
		if err != nil {
			xl.Error(err)
			return err
		}
		tarWriter := tar.NewWriter(tarFile)
		defer tarWriter.Close()
		for _, file := range filesToArchive {
			err := tarWriter.WriteHeader(&tar.Header{
				Name: path.Join(tarName, file.Name),
				Mode: 0600,
				Size: int64(len(file.Body)),
			})
			if err != nil {
				xl.Error(err)
				return err
			}
			_, err = tarWriter.Write([]byte(file.Body))
			if err != nil {
				xl.Error(err)
				return err
			}
		}

		xl.Infof("Archive %s success", fullTarName)
	}
	return
}

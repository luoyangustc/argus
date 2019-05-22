package mix

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"strconv"
	"strings"

	xlog "github.com/qiniu/xlog.v1"
	"qiniu.com/argus/com/uri"
	"qiniu.com/argus/com/util"
	"qiniu.com/argus/feature_group/distance"
)

type FaceSetConfig struct {
	NameFile   string
	GroupFile  string
	ImageDir   string
	FeatureDir string

	Threshold float32
	WorkSpace string `json:"workspace"`
}

type FaceSet struct {
	FaceSetConfig

	faces []struct {
		Name  string
		Group string
	}
	features []byte
}

func ScanTSV(
	ctx context.Context, filename string, b string,
	parse func(context.Context, []string) error,
) error {

	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		if len(line) == 0 {
			continue
		}
		strs := strings.Split(line, b)
		if err := parse(ctx, strs); err != nil {
			return err
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}

	return nil
}

func (s FaceSet) readGroupFile(ctx context.Context) (map[int]string, error) {
	var groups = map[int]string{}

	err := ScanTSV(ctx, s.GroupFile, " ",
		func(ctx context.Context, strs []string) error {
			xl := xlog.FromContextSafe(ctx)
			if len(strs) < 2 {
				xl.Warnf("%#v", strs)
				return errors.New(fmt.Sprintf("bad group line in %s", s.GroupFile))
			}
			index, _ := strconv.Atoi(strs[1])
			groups[index] = strs[0]
			return nil
		},
	)
	return groups, err
}

func (s FaceSet) readNameFile(ctx context.Context) (map[string]int, error) {
	var names = map[string]int{}
	err := ScanTSV(ctx, s.NameFile, "\t",
		func(_ context.Context, strs []string) error {
			xl := xlog.FromContextSafe(ctx)
			if len(strs) < 2 {
				xl.Warnf("%#v", strs)
				return errors.New(fmt.Sprintf("bad name line in %s", s.NameFile))
			}
			index, _ := strconv.Atoi(strs[1])
			names[strs[0]] = index
			return nil
		},
	)
	return names, err
}

func (s FaceSet) fetchFiles(ctx context.Context, url string) error {
	xl := xlog.FromContextSafe(ctx)
	local := path.Join(path.Join(s.WorkSpace, path.Base(url)))
	if strings.HasPrefix(url, "file://") {
		local = strings.TrimPrefix(url, "file://")
		_, err := os.Stat(local)
		if err != nil {
			return nil
		}
	}
	if strings.HasPrefix(url, "http://") {
		_ = os.MkdirAll(path.Dir(s.WorkSpace), 0755)
		c := uri.New(uri.WithHTTPHandler())
		resp, err := c.Get(ctx, uri.Request{URI: url})
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		stat, err := os.Stat(local)
		xl.Debug("stat", stat, err, resp.Size)
		if err == nil && stat.Size() == resp.Size {
			return nil
		}
		file, err := os.Create(local)
		if err != nil {
			return err
		}
		defer file.Close()
		n, err := io.Copy(file, resp.Body)
		xl.Debug("copy", n)
	}
	if strings.HasSuffix(local, ".tar") {
		_, err := util.ExtractTar(s.WorkSpace, local, false)
		if err != nil {
			return err
		}
	}
	return nil
}

func (s *FaceSet) Init(ctx context.Context) error {
	if len(s.FeatureDir) == 0 {
		return nil
	}

	groups, err := s.readGroupFile(ctx)
	if err != nil {
		return err
	}
	names, err := s.readNameFile(ctx)
	if err != nil {
		return err
	}

	dirs, err := ioutil.ReadDir(s.FeatureDir)
	if err != nil {
		return err
	}

	faces := []struct {
		Name  string
		Group string
	}{}
	features := make([]byte, 0)

	for _, dir := range dirs {
		var name = dir.Name()
		var group string
		if id, ok := names[name]; !ok {
			continue
		} else {
			group, ok = groups[id]
			if !ok {
				continue
			}
		}
		files, err := ioutil.ReadDir(path.Join(s.ImageDir, name))
		if err != nil {
			return err
		}
		for _, file := range files {
			feature, err := ioutil.ReadFile(path.Join(s.FeatureDir, name, file.Name()))
			if err != nil {
				return err
			}
			features = append(features, feature...)
			faces = append(faces, struct {
				Name  string
				Group string
			}{Name: name, Group: group})
		}
	}

	s.faces = faces
	s.features = features

	return nil
}

func (s *FaceSet) Reload(
	ctx context.Context, getFeature func(context.Context, []byte) ([]byte, error),
) error {

	var xl = xlog.FromContextSafe(ctx)
	_ = xl
	groups, err := s.readGroupFile(ctx)
	if err != nil {
		return err
	}
	names, err := s.readNameFile(ctx)
	if err != nil {
		return err
	}

	dirs, err := ioutil.ReadDir(s.ImageDir)
	if err != nil {
		return err
	}

	faces := []struct {
		Name  string
		Group string
	}{}
	features := make([]byte, 0)

	for _, dir := range dirs {
		var name = dir.Name()
		var group string
		if id, ok := names[name]; !ok {
			continue
		} else {
			group, ok = groups[id]
			if !ok {
				continue
			}
		}
		files, err := ioutil.ReadDir(path.Join(s.ImageDir, name))
		if err != nil {
			return err
		}
		for _, file := range files {
			img, err := ioutil.ReadFile(path.Join(s.ImageDir, name, file.Name()))
			if err != nil {
				return err
			}
			feature, err := getFeature(ctx, img)
			xl.Infof("load: %s %s %d %v", name, group, len(img), err)
			if err != nil {
				return err
			}
			if len(s.FeatureDir) > 0 {
				_ = os.MkdirAll(path.Join(s.FeatureDir, name), 0644)
				ioutil.WriteFile(path.Join(s.FeatureDir, name, file.Name()), feature, 0644)
			}
			features = append(features, feature...)
			faces = append(faces, struct {
				Name  string
				Group string
			}{Name: name, Group: group})
		}
	}

	s.faces = faces
	s.features = features

	return nil
}

func (s *FaceSet) Recognize(
	_ context.Context, pts [][]int, feature []byte,
) (ok bool, name, group string, score float32, err error) {

	fs := make([]float32, len(s.faces))
	distance.DistancesCosineCgoFlat(feature, s.features, fs)
	var max float32 = 0.0
	var index int = -1
	for i, d := range fs {
		if d > max {
			index = i
			max = d
		}
	}
	if max < s.Threshold || index == -1 {
		return
	}
	return true, s.faces[index].Name, s.faces[index].Group, max, nil
}

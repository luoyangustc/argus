package evaltest

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"math"

	"qiniu.com/argus/test/biz/env"
)

// v1/search/politician
type PoliticianResponse struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Result  struct {
		Review     bool `json:"review"`
		Detections []struct {
			BoundingBox struct {
				Pts   [][2]int `json:"pts"`
				Score float64  `json:"score"`
			} `json:"boundingBox"`
			Value struct {
				Name   string  `json:"name"`
				Group  string  `json:"group"`
				Score  float64 `json:"score"`
				Review bool    `json:"review"`
			} `json:"value"`
			Sample struct {
				Id  string   `json:"id"`
				Pts [][2]int `json:"Pts"`
				Url string   `json:"url"`
			} `json:"sample"`
		} `json:"detections"`
	} `json:"result"`
}

//##############################FEATURE##############################

type Feature struct {
	Index   int
	Group   string
	URL     string
	Feature []float32
	Pts     [4][2]int
	Size    string
}

type FeatureLib []Feature

type FeatureLibs map[string]FeatureLib

func GetFeature(filename string) FeatureLibs {
	var featureLarge FeatureLib
	var featureSmall FeatureLib
	features := FeatureLibs{"large": featureLarge, "small": featureSmall}
	var buf []byte
	var err error
	buf, err = env.Env.GetTSV(filename)
	if err != nil {
		panic(err)
	}

	reader := bufio.NewReader(bytes.NewReader(buf))

	scanner := bufio.NewScanner(reader)
	scanner.Split(bufio.ScanLines)
	for scanner.Scan() {
		var feature Feature
		line1 := scanner.Bytes()
		if err = json.Unmarshal(line1, &feature); err != nil {
			panic(err)
		}

		features[feature.Size] = append(features[feature.Size], feature)

	}
	return features
}

func MaxFeature(features FeatureLib, feature []float32) (float64, Feature) {
	max := 0.0
	idx := -1
	for i, fea := range features {
		score, _ := Cosine(fea.Feature, feature)
		if score > max {
			max = score
			idx = i
		}
	}
	return max, features[idx]
}

func Cosine(a []float32, b []float32) (cosine float64, err error) {
	count := 0
	lengthA := len(a)
	lengthB := len(b)
	if lengthA > lengthB {
		count = lengthA
	} else {
		count = lengthB
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		ak := float64(a[k])
		bk := float64(b[k])
		if k >= lengthA {
			s2 += math.Pow(bk, 2)
			continue
		}
		if k >= lengthB {
			s1 += math.Pow(ak, 2)
			continue
		}
		sumA += ak * bk
		s1 += math.Pow(ak, 2)
		s2 += math.Pow(bk, 2)
	}
	if s1 <= 0.000001 || s2 <= 0.000001 {
		return 0.0, errors.New("Vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}

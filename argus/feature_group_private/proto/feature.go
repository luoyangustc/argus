package proto

import (
	"encoding/base64"
	"encoding/json"
)

type FeatureValue []byte
type FeatureID string
type FeatureTag string
type FeatureHashKey uint32

// HashKeyRange : [range[0], range[1])
type HashKeyRange [2]FeatureHashKey
type BoundingBoxPts [][2]int
type BoundingBoxScore float32
type BoundingBox struct {
	Pts   BoundingBoxPts   `json:"pts,omitempty" bson:"pts,omitempty"`
	Score BoundingBoxScore `json:"score,omitempty" bson:"score,omitempty"`
}

func (f FeatureValue) ToFeatureValueJson() FeatureValueJson {
	return FeatureValueJson(base64.StdEncoding.EncodeToString(f))
}

// Feature通用定义，含bson结构，可直接mgo接口读写
type Feature struct {
	ID      FeatureID       `bson:"id"`
	Value   FeatureValue    `bson:"value"`
	Tag     FeatureTag      `bson:"tag"`
	Desc    json.RawMessage `bson:"desc"`
	Group   GroupName       `bson:"group"`
	HashKey FeatureHashKey  `bson:"hash_key,omitempty"`

	// face meta
	BoundingBox BoundingBox `bson:"bounding_box,omitempty"`
	FaceQuality FaceQuality `bson:"face_quality,omitempty"`
}

func (f Feature) ToImage() Image {
	return Image{
		ID:   f.ID,
		Tag:  f.Tag,
		Desc: json.RawMessage(f.Desc),
	}
}

func (f Feature) ToFeatureJson() FeatureJson {
	return FeatureJson{
		ID:          f.ID,
		Value:       f.Value.ToFeatureValueJson(),
		Tag:         f.Tag,
		Desc:        f.Desc,
		BoundingBox: f.BoundingBox,
		//TODO output face_quality
	}
}

////////////////////////////////////////////////////////////////////////////////

type FeatureValueJson string

func (fj FeatureValueJson) ToFeatureValue() FeatureValue {
	bs, _ := base64.StdEncoding.DecodeString(string(fj))
	return FeatureValue(bs)
}

type FeatureJson struct {
	ID          FeatureID        `json:"id,omitempty"`
	Value       FeatureValueJson `json:"value,omitempty"`
	Tag         FeatureTag       `json:"tag,omitempty"`
	Desc        json.RawMessage  `json:"desc,omitempty"`
	BoundingBox BoundingBox      `json:"bounding_box,omitempty"`
	FaceQuality *FaceQuality     `json:"face_quality,omitempty"`
}

func (fj FeatureJson) ToFeature() Feature {
	feature := Feature{
		ID:          fj.ID,
		Value:       fj.Value.ToFeatureValue(),
		Tag:         fj.Tag,
		Desc:        fj.Desc,
		BoundingBox: fj.BoundingBox,
	}
	if fj.FaceQuality != nil {
		feature.FaceQuality = *fj.FaceQuality
	}
	return feature
}

package proto

import (
	"encoding/json"
)

type ImageURI string

type Image struct {
	ID          FeatureID       `json:"id,omitempty"`
	URI         ImageURI        `json:"uri,omitempty"`
	Tag         FeatureTag      `json:"tag,omitempty"`
	Desc        json.RawMessage `json:"desc,omitempty"`
	BoundingBox BoundingBox     `json:"bounding_box,omitempty"`
}

type ImageAttribute struct {
	BoundingBoxes []BoundingBox `json:"bounding_boxes,omitempty"`
}

type Data struct {
	URI       ImageURI       `json:"uri"`
	Attribute ImageAttribute `json:"attribute,omitempty"`
}

type ImageExtend struct {
	Data   []Data `json:"data"`
	Params struct {
		Threshold  float32 `json:"threshold"`
		Limit      int     `json:"limit"`
		UseQuality bool    `json:"use_quality,omitempty"`
	} `json:"params,omitempty"`
}

func (i Image) ToFeature() Feature {
	return Feature{
		ID:    i.ID,
		Value: FeatureValue(i.URI),
		Tag:   i.Tag,
		Desc:  i.Desc,
	}
}

func (i Image) ToImageJson() ImageJson {
	return ImageJson{
		ID:   i.ID,
		URI:  ImageURI(i.URI),
		Tag:  i.Tag,
		Desc: i.Desc,
	}
}

type ImageJson struct {
	ID   FeatureID       `json:"id,omitempty"`
	URI  ImageURI        `json:"uri,omitempty"`
	Tag  FeatureTag      `json:"tag,omitempty"`
	Desc json.RawMessage `json:"desc,omitempty"`
}

func (i ImageJson) ToImage() Image {
	return Image{
		ID:   i.ID,
		URI:  ImageURI(i.URI),
		Tag:  i.Tag,
		Desc: i.Desc,
	}
}

package proto

import "testing"

func TestProto(t *testing.T) {
	ImageJson{}.ToImage()
	Image{}.ToFeature()
	Image{}.ToImageJson()
	Feature{}.ToImage()
	Feature{}.ToFeatureJson()
	FeatureValue{}.ToFeatureValueJson()
	var i FeatureValueJson
	i.ToFeatureValue()
	FeatureJson{}.ToFeature()
}

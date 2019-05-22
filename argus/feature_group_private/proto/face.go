package proto

// 人脸朝向 - 八分类
type FaceOrientation string

const (
	FaceOrientationUp        FaceOrientation = "up"
	FaceOrientationUpLeft    FaceOrientation = "up_left"
	FaceOrientationLeft      FaceOrientation = "left"
	FaceOrientationDownLeft  FaceOrientation = "down_left"
	FaceOrientationDown      FaceOrientation = "down"
	FaceOrientationDownRight FaceOrientation = "down_right"
	FaceOrientationRight     FaceOrientation = "right"
	FaceOrientationUpRight   FaceOrientation = "up_right"
)

// 人脸质量 - 由facex-detect返回， 五分类
type FaceQualityClass string

const (
	FaceQualityClear    FaceQualityClass = "clear"
	FaceQualityBlur     FaceQualityClass = "blur"
	FaceQualityNegative FaceQualityClass = "neg"
	FaceQualityCover    FaceQualityClass = "cover"
	FaceQualityPose     FaceQualityClass = "pose"
	FaceQualitySmall    FaceQualityClass = "small"
)

type FaceQuality struct {
	Quality      FaceQualityClass             `json:"quality,omitempty" bson:"quality,omitempty"`
	Orientation  FaceOrientation              `json:"orientation,omitempty" bson:"orientation,omitempty"`
	QualityScore map[FaceQualityClass]float32 `json:"-" bson:"quality_score,omitempty"`
}

type FaceDetectBox struct {
	BoundingBox BoundingBox
	Quality     FaceQuality
}

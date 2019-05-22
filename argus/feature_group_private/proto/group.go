package proto

//--------------------------- Group ---------------------------//
type GroupName string

const (
	GroupUnknown int = iota
	GroupCreated
	GroupInitialized
)

type GroupConfig struct {
	Dimension int    `json:"dimension" bson:"dimension"`
	Precision int    `json:"precision" bson:"precision"`
	Capacity  int    `json:"capacity" bson:"capacity"`
	Version   uint64 `json:"version" bson:"version"`
}

type GroupTagInfo struct {
	Name  FeatureTag `json:"name" bson:"_id"`
	Count int        `json:"count" bson:"count"`
}

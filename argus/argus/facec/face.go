package facec

import "time"

// CanDetectFace judge the face can be detected
// bbox format: [[leftup.x,leftup.y], [rightup.x, rightup.y], [rightdown.x, rightdown.y],[leftdown.x, leftdown.y]]
// axis
//    0  ->   1
//    +---------------------> x
//   0|
//    |
//   ||
//   v|
//    |
//   1|
//    |
//    v
//    y
func CanDetectFace(bbox [][]int64) bool {
	leftUp, rightUp, rightDown := bbox[0], bbox[1], bbox[2]
	longEdge, shortEdge := rightDown[1]-rightUp[1], rightUp[0]-leftUp[0]

	if longEdge < shortEdge {
		longEdge, shortEdge = shortEdge, longEdge
	}

	if longEdge < 80 {
		return false
	}

	if float64(longEdge)/float64(shortEdge) > 1.5 {
		return false
	}

	return true
}

const (
	// GroupStatusUpdating group is updating
	GroupStatusUpdating = 1
	// GroupStatusDone group is updated
	GroupStatusDone = 2
	//VersionFormat agreed version format
	VersionFormat = "2006-01-02 15:04:05"
)

// NewVersion create the vesion value
func NewVersion() string {
	return time.Now().Format(VersionFormat)
}

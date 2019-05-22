package facec

// UnknownClusterID the initialization cluster id
const UnknownClusterID = -2

// UnknownGtID the initialization gt id
const UnknownGtID = -1

func calcGroupID(clusterID, groupID int64) int64 {
	if groupID != UnknownGtID {
		return groupID
	}
	if clusterID != UnknownClusterID {
		return clusterID
	}

	return UnknownClusterID
}

//----------------------------------------------------------------------------//

const (
	_DataURIPrefix = "data:application/octet-stream;base64,"
)

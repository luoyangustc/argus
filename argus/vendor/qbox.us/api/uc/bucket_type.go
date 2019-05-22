package uc

type BucketType int

const (
	TYPE_COM BucketType = iota
	TYPE_MEDIA
	TYPE_DL
)

func (v BucketType) Valid() bool {
	switch v {
	case TYPE_COM, TYPE_MEDIA, TYPE_DL:
		return true
	}
	return false
}

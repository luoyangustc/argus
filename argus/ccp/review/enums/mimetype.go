package enums

// MimeType define
type MimeType string

const (
	MimeTypeImage MimeType = "IMAGE"
	MimeTypeVideo MimeType = "VIDEO"
	MimeTypeLive  MimeType = "LIVE"
)

func (this MimeType) IsValid() bool {
	switch this {
	case MimeTypeImage, MimeTypeVideo, MimeTypeLive:
		return true
	default:
		return false
	}
}

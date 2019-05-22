package enums

type MimeType string

const (
	MimeTypeImage MimeType = "image"
	MimeTypeVideo MimeType = "video"
	MimeTypeLive  MimeType = "live"
)

func (this MimeType) IsValid() bool {
	switch this {
	case MimeTypeImage, MimeTypeVideo, MimeTypeLive:
		return true
	default:
		return false
	}
}

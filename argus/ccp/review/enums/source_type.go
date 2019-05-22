package enums

// SourceType define
type SourceType string

const (
	SourceTypeKodo SourceType = "KODO"
	SourceTypeApi  SourceType = "API"
)

func (this SourceType) IsValid() bool {
	switch this {
	case SourceTypeKodo, SourceTypeApi:
		return true
	default:
		return false
	}
}

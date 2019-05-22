package rpc

//import "strings"

// --------------------------------------------------------------------

type ErrorRet struct {
	Error string `json:"error"`
}

/*
func GetShortError(err error) string {
	str := err.Error()
	pos := strings.LastIndex(str, ":")
	if pos < 0 {
		return str
	}
	if str[pos+1] == ' ' {
		pos++
	}
	return str[pos+1:]
}

func Error(err error) interface{} {
	return ErrorRet{GetShortError(err)}
}
*/
// --------------------------------------------------------------------

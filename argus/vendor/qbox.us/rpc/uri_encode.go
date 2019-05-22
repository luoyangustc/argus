package rpc

import "qbox.us/net/uri"

// URL:
//	 http://host/url
//	 https://host/url
// Path:
//	 AbsolutePath	(Must start with '/')
//	 Pid:RelPath	(Pid.len = 16)
//	 Id: 			(Id.len = 16)
//	 :LinkId:RelPath
//	 :LinkId
func EncodeURI(u string) string {
	return uri.Encode(u)
}

func DecodeURI(encodedURI string) (u string, err error) {
	return uri.Decode(encodedURI)
}

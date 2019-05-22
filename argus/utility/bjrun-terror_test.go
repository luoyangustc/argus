package utility

// import (
// 	"testing"
// )

// func TestBjRTerror(t *testing.T) {
// 	service, ctx := getMockContext(t)
// 	service.terrorClassify = mockTerrorClassify{}
// 	service.terrorDetect = mockTerrorDetect{}
// 	ctx.Exec(`
// 		post http://argus.ava.ai/v1/bjrun/terror
// 		auth |authstub -uid 1 -utype 4|
// 		json '{
// 			"data": {
// 				"uri": "http://test.image.jpg"
// 			}
// 		}'
// 		ret 200
// 		header Content-Type $(mime)
// 		equal $(mime) 'application/json'
// 		echo $(resp.body)
// 		json '{
// 			"code": $(code),
// 			"result":{
// 					   "label":$(label),
// 					   "score":$(score),
// 					   "class":$(class),
// 					   "review":$(review)
// 			}
// 		}'
// 		equal $(code) 0
// 		equal $(label) 1
// 		equal $(score) 0.97
// 		equal $(review) false

// 	`)
// }

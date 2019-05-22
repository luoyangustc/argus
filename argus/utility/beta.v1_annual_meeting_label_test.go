package utility

import (
	"testing"
)

func TestAnnualMeetingLabel(t *testing.T) {
	service, ctx := getMockContext(t)
	service.iFaceDetect = mockFaceDetect{}
	service.iBluedD = mockBluedD{}
	ctx.Exec(`
		post http://argus.ava.ai/v1/beta/human/label
		auth |authstub -uid 1 -utype 4|
		json '{
			"data": {
				"uri": "http://test.image.jpg"  
			}   
		}'
		ret 200
		header Content-Type $(mime) 
		equal $(mime) 'application/json'
		echo $(resp.body)
		json '{
			"code": $(code),
			"result":{
				"tags": $(tags)	
			}
		}'
		let $(tgs) ["校花校草我爱你"]
		equal $(code) 0
		equalSet $(tags) $(tgs)
	`)
}

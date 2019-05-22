package proto

// https://github.com/qbox/ava/blob/dev/docs/AtServing.api.md#ocr-detect

// OcrParams ... ocr params
type OcrParams struct {
	ImageType string `json:"image_type"`
}

// ####################################################
// ocr detect start

// OcrDetectRes ... ocr detect res
type OcrDetectRes struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Result  OcrDetectResult `json:"result"`
}

// OcrDetectResult ... ocr detect result
type OcrDetectResult struct {
	AreaRatio float64  `json:"area_ratio"`
	Bboxes    [][4]int `json:"bboxes"`
	ImgType   string   `json:"img_type"`
}

// NewOcrDetectParams ... new ocr detect params
func NewOcrDetectParams(str string) OcrParams {
	var para OcrParams
	para.ImageType = str
	return para
}

// ocr detect end
// ####################################################

// ####################################################
// ocr recognize start

// OcrRecognizeParams ... ocr recognize params
type OcrRecognizeParams struct {
	ImageType string   `json:"image_type"`
	Bboxes    [][4]int `json:"bboxes"`
}

// OcrRecognizeRes ... ocr recognize res
type OcrRecognizeRes struct {
	Code    int                `json:"code"`
	Message string             `json:"message"`
	Result  OcrRecognizeResult `json:"result"`
}

// OcrRecognizeResult ... ocr recognize result
type OcrRecognizeResult struct {
	Bboxes [][4][2]int `json:"bboxes"`
	Texts  []string    `json:"texts"`
}

// NewOcrRecognizeParams ... new ocr recognize params
func NewOcrRecognizeParams(imageType string, bboxes [][4]int) OcrRecognizeParams {
	var para OcrRecognizeParams
	para.ImageType = imageType
	para.Bboxes = bboxes
	return para
}

// ocr recognize end
// ####################################################

// ####################################################
// ocr sari start

// OcrSariCranParams ... ocr sari-crann params
type OcrSariCranParams struct {
	Bboxes [][4][2]int `json:"bboxes"`
}

// NewOcrSariCrannParams .. new ocr sari-crann params
func NewOcrSariCrannParams(bboxes [][4][2]int) *OcrSariCranParams {
	return &OcrSariCranParams{
		Bboxes: bboxes,
	}
}

// OcrSariCranRes ... ocr sari-crann res
type OcrSariCranRes struct {
	Code    int               `json:"code"`
	Message string            `json:"message"`
	Result  OcrSariCranResult `json:"result"`
}

// OcrSariCranResult ... ocr sari-crann result
type OcrSariCranResult struct {
	Text []string `json:"text"`
}

// OcrSariIDPreParams ... ocr-sari-id-pre params
type OcrSariIDPreParams struct {
	Type          string                 `json:"type,omitempty"`
	Bboxes        [][4][2]int            `json:"bboxes,omitempty"`
	Class         int                    `json:"class"`
	Texts         []string               `json:"texts,omitempty"`
	Names         []string               `json:"names,omitempty"`
	Regions       [][4][2]int            `json:"regions,omitempty"`
	DetectedBoxes [][4][2]int            `json:"detectedBoxes,omitempty"`
	AlignedImg    string                 `json:"alignedImg,omitempty"`
	Res           map[string]interface{} `json:"res,omitempty"`
}

// OcrSariIDPreParams2 ... ocr-sari-id-pre params
type OcrSariIDPreParams2 struct {
	Type          string      `json:"type,omitempty"`
	Bboxes        [][4][2]int `json:"bboxes,omitempty"`
	Class         int         `json:"class,omitempty"`
	Texts         []string    `json:"texts,omitempty"`
	Names         []string    `json:"names,omitempty"`
	Regions       [][4][2]int `json:"regions,omitempty"`
	DetectedBoxes [][4][2]int `json:"detectedBoxes,omitempty"`
	AlignedImg    string      `json:"alignedImg,omitempty"`
	Res           [][2]string `json:"res,omitempty"`
}

// NewOcrSariIDPredetectParams ... new ocr-sari-id-predetect params
func NewOcrSariIDPredetectParams(types string) OcrSariIDPreParams2 {
	return OcrSariIDPreParams2{
		Type: types,
	}
}

// OcrSariIDPreRes ... ocr sari-id-pre res
type OcrSariIDPreRes struct {
	Code    int                `json:"code"`
	Message string             `json:"message"`
	Result  OcrSariIDPreParams `json:"result"`
}

// OcrSariIDCardResult ... ocr sari-id-card result
type OcrSariIDCardResult struct {
	URI    string            `json:"uri"`
	Bboxes [][4][2]int       `json:"bboxes"`
	Type   int               `json:"type"`
	Res    map[string]string `json:"res"`
}

// OcrSariIDCardRes ... ocr sari-id-card res
type OcrSariIDCardRes struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  OcrSariIDCardResult `json:"result"`
}

// ocr sari end
// ####################################################

// ####################################################
// ocr sence recognize start

// OcrSenceRecogText ... text of result of ocr sence recognize
type OcrSenceRecogText struct {
	Bboxes [8]int `json:"bboxes"`
	Text   string `json:"text"`
}

// OcrSenceRecogResult ... result of ocr sence recognize
type OcrSenceRecogResult struct {
	Texts []OcrSenceRecogText `json:"texts"`
}

// OcrSenceDetectResult ... result of ocr sence detect
type OcrSenceDetectResult struct {
	Bboxes [][8]int `json:"bboxes"`
}

// OcrSenceDetectRes ... response of ocr sence detect
type OcrSenceDetectRes struct {
	Code    int                  `json:"code"`
	Message string               `json:"message"`
	Result  OcrSenceDetectResult `json:"result"`
}

// OcrSenceRecogDataRes ... response of ocr sence recognize
type OcrSenceRecogDataRes struct {
	Code    int                 `json:"code"`
	Message string              `json:"message"`
	Result  OcrSenceRecogResult `json:"result"`
}

// ArgOcrTextResult ... result of argus-ocr-text result
type ArgOcrTextResult struct {
	Type   string      `json:"type"`
	Bboxes [][4][2]int `json:"bboxes"`
	Texts  []string    `json:"texts"`
}

// ArgOcrTextRes ... response of argus-ocr-text
type ArgOcrTextRes struct {
	Code    int              `json:"code"`
	Message string           `json:"message"`
	Result  ArgOcrTextResult `json:"result"`
}

// ocr sence recognizeri end
// ####################################################

// ArgOcrSence ...
type ArgOcrSence struct {
	Bboxes [][8]int `json:"bboxes"`
	Text   []string `json:"text"`
}

// ArgOcrSenceRes ...
type ArgOcrSenceRes struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Result  ArgOcrSence `json:"result"`
}

// ArgOcrSenceError ...
type ArgOcrSenceError struct {
	Error string `json:"error"`
}

// OcrCtpnRes ... ocr ctpn res
type OcrCtpnRes struct {
	Code    int           `json:"code"`
	Message string        `json:"message"`
	Result  OcrCtpnResult `json:"result"`
}

// OcrCtpnResult ... ocr ctpn result
type OcrCtpnResult struct {
	Bboxes [][4][2]int `json:"bboxes"`
}

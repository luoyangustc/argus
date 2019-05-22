package biz

func newFloat32(f float32) *float32 { return &f }

func v(v interface{}) func() interface{} { return func() interface{} { return v } }
func co(c bool, f1, f2 func() interface{}) interface{} {
	if c {
		return f1()
	} else {
		return f2()
	}
}

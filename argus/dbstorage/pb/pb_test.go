package pb

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func Test_Pb(t *testing.T) {
	bar := StartNew(10)
	for i := 0; i < 10; i++ {
		bar.Increment()
	}
	bar.FinishPrint("End")

	count := bar.Get()
	assert.Equal(t, 10, int(count))
}

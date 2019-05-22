package facec

import (
	"testing"
)

func TestCanDetectFace(t *testing.T) {
	if CanDetectFace([][]int64{
		{0, 0},
		{4, 0},
		{4, 4},
		{0, 4},
	}) {
		t.Fatal("error")
	}

	if CanDetectFace([][]int64{
		{0, 0},
		{4, 0},
		{4, 2},
		{0, 2},
	}) {
		t.Fatal("error")
	}

	if CanDetectFace([][]int64{
		{0, 0},
		{400, 0},
		{400, 2},
		{0, 2},
	}) {
		t.Fatal("error")
	}

	if !CanDetectFace([][]int64{
		{0, 0},
		{500, 0},
		{500, 400},
		{0, 400},
	}) {
		t.Fatal("error")
	}

	if !CanDetectFace([][]int64{
		{0, 0},
		{300, 0},
		{300, 400},
		{0, 400},
	}) {
		t.Fatal("error")
	}
}

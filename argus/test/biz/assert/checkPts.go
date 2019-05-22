package assert

import (
	. "github.com/onsi/gomega"
)

func CheckPts(pts [][2]int) {
	Expect(pts[0][0]).Should(Equal(pts[3][0]))
	Expect(pts[0][1]).Should(Equal(pts[1][1]))
	Expect(pts[1][0]).Should(Equal(pts[2][0]))
	Expect(pts[2][1]).Should(Equal(pts[3][1]))
	Expect(pts[0][0]).Should(BeNumerically("<", pts[1][0]))
	Expect(pts[0][1]).Should(BeNumerically("<", pts[3][1]))
}

func CheckOcrPts(pts [4][2]int) {
	Expect(pts[0][0]).Should(Equal(pts[3][0]))
	Expect(pts[0][1]).Should(Equal(pts[1][1]))
	Expect(pts[1][0]).Should(Equal(pts[2][0]))
	Expect(pts[2][1]).Should(Equal(pts[3][1]))
	Expect(pts[0][0]).Should(BeNumerically("<=", pts[1][0]))
	Expect(pts[0][1]).Should(BeNumerically("<=", pts[3][1]))
}

func FindListStr(target string, list []string) (i int) {
	i = -1
	for j, v := range list {
		if v == target {
			i = j
			return
		}
	}
	return
}

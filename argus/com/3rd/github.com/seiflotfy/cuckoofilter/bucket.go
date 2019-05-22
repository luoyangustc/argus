package cuckoofilter

type bucket [bucketSize]FingerprintType

const (
	nullFp     = FingerprintType(0)
	bucketSize = 4
)

func (b *bucket) insert(fp FingerprintType) bool {
	for i, tfp := range b {
		if tfp == nullFp {
			b[i] = fp
			return true
		}
	}
	return false
}

func (b *bucket) delete(fp FingerprintType) bool {
	for i, tfp := range b {
		if tfp == fp {
			b[i] = nullFp
			return true
		}
	}
	return false
}

func (b *bucket) getFingerprintIndex(fp FingerprintType) int {
	for i, tfp := range b {
		if tfp == fp {
			return i
		}
	}
	return -1
}

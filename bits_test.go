package leetcode

import (
	"fmt"
	"testing"
)

func TestHammingWeight(t *testing.T) {
	fmt.Println(HammingWeight(11))
}

func TestIsPowerOfTwo(t *testing.T) {
	t.Log(isPowerOfTwo(6))
}

func Test_hammingDistance(t *testing.T) {
	t.Log(hammingDistance(1, 4))
}

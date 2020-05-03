package leetcode

import (
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestSortArray(t *testing.T) {
	type test struct {
		input []int
		want  []int
	}
	tests := []test{
		{[]int{6, 1, 2, 7, 9, 3, 4, 5, 10, 8}, []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}},
		{[]int{6, 1, 2, 7, 9}, []int{1, 2, 6, 7, 9}},
		{[]int{5, 1, 1, 2, 0, 0}, []int{0, 0, 1, 1, 2, 5}},
		{[]int{1}, []int{1}},
		{[]int{1, 2, 3}, []int{1, 2, 3}},
	}
	for _, tc := range tests {
		got := SortArray3(tc.input)
		if !reflect.DeepEqual(tc.want, got) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestSortArray2(t *testing.T) {
	// nums := []int{6, 1, 2, 7, 9, 3, 4, 5, 10, 8}
	// nums := []int{6, 1, 2, 7, 9}
	nums := []int{5, 1, 1, 2, 0, 0}
	// nums := []int{1}
	ShellSort(nums)
	t.Log(nums)
}

func Test_quickSort3Way(t *testing.T) {
	// nums := []int{5, 1, 1, 2, 0, 0}
	nums := []int{6, 1, 2, 7, 9, 3, 4, 5, 10, 8}
	QuickSort3Way(nums, 0, 9)
	t.Log(nums)
}

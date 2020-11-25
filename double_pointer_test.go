package leetcode

import (
	"testing"
)

func TestMinSubArrayLen(t *testing.T) {
	type args struct {
		s    int
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{7, []int{2, 3, 1, 2, 4, 3}}, 2},
		{"normal2", args{15, []int{5, 1, 3, 5, 10, 7, 4, 9, 2, 8}}, 2},
		{"no result", args{3, []int{1, 1}}, 0},
		{"empty array", args{3, []int{}}, 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MinSubArrayLen(tt.args.s, tt.args.nums); got != tt.want {
				t.Errorf("MinSubArrayLen() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSearchMatrix(t *testing.T) {
	// type args struct {
	// 	matrix [][]int
	// 	target int
	// }
	// tests := []struct {
	// 	name string
	// 	args args
	// 	want bool
	// }{
	// // TODO: Add test cases.
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		if got := SearchMatrix(tt.args.matrix, tt.args.target); got != tt.want {
	// 			t.Errorf("SearchMatrix() = %v, want %v", got, tt.want)
	// 		}
	// 	})
	// }
	matrix := [][]int{{1, 4, 7, 11, 15}, {2, 5, 8, 12, 19}, {3, 6, 9, 16, 22}, {10, 13, 14, 17, 24}, {18, 21, 23, 26, 30}}
	got := SearchMatrix2(matrix, 15)
	t.Log(got)
}

func TestFindDuplicate(t *testing.T) {
	// nums := []int{3, 1, 3, 4, 2}
	nums := []int{1, 3, 4, 2, 2}
	t.Log(FindDuplicate(nums))
}

func Test_search2(t *testing.T) {
	t.Log(search2([]int{2, 5, 6, 0, 0, 1, 2}, 0) == true)
	t.Log(search2([]int{2, 5, 6, 0, 0, 1, 2}, 3) == false)
	t.Log(search2([]int{1, 3, 1, 1, 1}, 3) == true)
}

func Test_threeSum(t *testing.T) {
	nums := []int{-1, 0, 1, 2, -1, -4}
	t.Log(threeSum(nums))
	nums2 := []int{-4, -2, 1, -5, -4, -4, 4, -2, 0, 4, 0, -2, 3, 1, -5, 0}
	t.Log(threeSum(nums2))
}

func Test_threeSumClosest(t *testing.T) {
	nums := []int{-1, 2, 1, -4}
	target := 1

	nums = []int{1, 2, 4, 8, 16, 32, 64, 128}
	target = 82
	t.Log(threeSumClosest(nums, target))
}

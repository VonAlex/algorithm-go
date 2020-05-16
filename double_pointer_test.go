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

func TestBinarySearch(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name      string
		args      args
		wantIndex int
	}{
		{"case one num", args{[]int{1}, 1}, 0},
		{"case repeat num", args{[]int{1, 2, 2, 3, 4}, 2}, 2},
		{"case  normal", args{[]int{1, 2, 2, 3, 4}, 3}, 3},
		{"case no target", args{[]int{1, 2, 2, 3, 4}, 5}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotIndex := BinarySearch(tt.args.nums, tt.args.target); gotIndex != tt.wantIndex {
				t.Errorf("BinarySearch() = %v, want %v", gotIndex, tt.wantIndex)
			}
		})
	}
}

func TestFindMin(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"sorted", args{[]int{1, 2, 3, 4}}, 1},
		{"one item", args{[]int{1}}, 1},
		{"two item", args{[]int{3, 2}}, 2},
		{"normal", args{[]int{3, 4, 5, 1, 2}}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := FindMin(tt.args.nums); got != tt.want {
				t.Errorf("FindMin() = %v, want %v", got, tt.want)
			}
		})
	}
}

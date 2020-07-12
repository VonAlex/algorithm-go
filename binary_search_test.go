package leetcode

import (
	"testing"
)

func Test_binarySearch(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"case 1", args{[]int{5}, 5}, 0},
		{"case 2", args{[]int{1, 2, 3, 4, 5}, 5}, 4},
		{"case 3", args{[]int{1, 2, 3, 4, 5}, 6}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := binarySearch(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("binarySearch() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_binarySearchFindLeftBound(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"case 1", args{[]int{5}, 5}, 0},
		{"case 2", args{[]int{1, 2, 2, 2, 5}, 2}, 1},
		{"case 3", args{[]int{1, 2, 3, 4, 5}, 6}, -1},
		{"case 4", args{[]int{2, 2, 3, 4, 5}, 1}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := binarySearchFindLeftBound(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("binarySearchFindLeftBound() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_binarySearchFindRightBound(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"case 1", args{[]int{5}, 5}, 0},
		{"case 2", args{[]int{1, 2, 2, 2, 5}, 2}, 3},
		{"case 3", args{[]int{1, 2, 3, 4, 5}, 6}, -1},
		{"case 4", args{[]int{2, 2, 3, 4, 5}, 1}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := binarySearchFindRightBound(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("binarySearchFindRightBound() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_search(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"case1", args{[]int{3, 1}, 1}, 1},
		{"case2", args{[]int{4, 5, 6, 7, 0, 1, 2}, 0}, 4},
		{"case3", args{[]int{4, 5, 6, 7, 0, 1, 2}, 2}, 6},
		{"case4", args{[]int{4, 5, 6, 7, 0, 1, 2}, 3}, -1},
		{"case5", args{[]int{4, 5, 6, 7, 0, 1, 2}, 8}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := search(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("search() = %v, want %v", got, tt.want)
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
			if got := findMin(tt.args.nums); got != tt.want {
				t.Errorf("FindMin() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestFindMin2(t *testing.T) {
	// type args struct {
	// 	nums []int
	// }
	// tests := []struct {
	// 	name string
	// 	args args
	// 	want int
	// }{
	// 	{"case1", args{[]int{2, 2, 2, 0, 1}}, 0},
	// 	{"case2", args{[]int{3, 1, 3}}, 1},
	// 	{"case3", args{[]int{1, 1}}, 1},
	// 	{"case4", args{[]int{4, 4, 5, 6, 7, 1, 2, 4, 4}}, 1},
	// 	{"case5", args{[]int{3}}, 3},
	// }
	// for _, tt := range tests {
	// 	t.Run(tt.name, func(t *testing.T) {
	// 		if got := findMin3(tt.args.nums); got != tt.want {
	// 			t.Errorf("FindMin() = %v, want %v", got, tt.want)
	// 		}
	// 	})
	// }

	t.Log(findMin3([]int{4, 4, 5, 6, 7, 1, 2, 4, 4}))
}

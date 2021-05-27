package leetcode

import (
	"reflect"
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
		{"case 0", args{[]int{}, 5}, -1},
		{"case 1", args{[]int{5}, 5}, 0},
		{"case 2", args{[]int{1, 2, 2, 2, 5}, 2}, 1},
		{"case 3", args{[]int{1, 2, 3, 4, 5}, 6}, -1},
		{"case 4", args{[]int{2, 2, 3, 4, 5}, 1}, -1},
		{"case 5", args{[]int{2, 2, 3, 4, 8}, 5}, -1},
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
		{"case5", args{[]int{}, 8}, -1},
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
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"case1", args{[]int{2, 2, 2, 0, 1}}, 0},
		{"case2", args{[]int{3, 1, 3}}, 1},
		{"case3", args{[]int{1, 1}}, 1},
		{"case4", args{[]int{4, 4, 5, 6, 7, 1, 2, 4, 4}}, 1},
		{"case5", args{[]int{3}}, 3},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := findMin2(tt.args.nums); got != tt.want {
				t.Errorf("FindMin() = %v, want %v", got, tt.want)
			}
		})
	}

	// t.Log(findMin2([]int{4, 4, 5, 6, 7, 1, 2, 4, 4}))
}

func Test_searchRange(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"emtpy", args{[]int{}, 0}, []int{-1, -1}},
		{"one", args{[]int{1}, 1}, []int{0, 0}},
		{"not find", args{[]int{5, 7, 7, 8, 8, 10}, 6}, []int{-1, -1}},
		{"find", args{[]int{5, 7, 7, 8, 8, 10}, 8}, []int{3, 4}},
		{"left", args{[]int{5, 7, 7, 8, 8, 10}, 4}, []int{-1, -1}},
		{"right", args{[]int{5, 7, 7, 8, 8, 10}, 11}, []int{-1, -1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := searchRange(tt.args.nums, tt.args.target); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("searchRange() = %v, want %v", got, tt.want)
			}
		})
	}
}

package leetcode

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
)

func TestTwo3Sum(t *testing.T) {
	type test struct {
		input  []int
		target int
		want   []int
	}

	tests := []test{
		{input: []int{2, 7, 11, 15}, target: 9, want: []int{0, 1}},
		{input: []int{2, 7, 11, 15}, target: 20, want: []int{0, 0}},
		{input: []int{3, 2, 4}, target: 6, want: []int{1, 2}},
	}

	for _, tc := range tests {
		got := TwoSum2(tc.input, tc.target)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestFindRepeatNumber(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{input: []int{2, 3, 1, 0, 2, 5, 3}, want: 2},
		{input: []int{2, 3, 1, 0}, want: -1},
	}
	for _, tc := range tests {
		got := FindRepeatNumber2(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestDistributeCandies(t *testing.T) {
	type test struct {
		candies int
		people  int
		want    []int
	}
	tests := []test{
		{candies: 7, people: 4, want: []int{1, 2, 3, 1}},
		{candies: 10, people: 3, want: []int{5, 2, 3}},
	}
	for _, tc := range tests {
		got := DistributeCandies(tc.candies, tc.people)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestExange(t *testing.T) {
	nums := []int{1, 2, 3, 4, 5}
	t.Log(Exchange(nums))
}

func TestPivotIndex(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{1, 7, 3, 6, 5, 6}, 3},
		{[]int{1, 2, 3}, -1},
		{[]int{}, -1},
	}
	for _, tc := range tests {
		got := PivotIndex(tc.input)
		if got != tc.want {
			t.Log(tc.want, got)
		}
	}
}

func TestDominantIndex(t *testing.T) {
	nums := []int{0, 0, 0, 1}
	t.Log(DominantIndex2(nums))
}

func TestPlusOne(t *testing.T) {
	type test struct {
		input []int
		want  []int
	}
	tests := []test{
		{[]int{9}, []int{1, 0}}, // 注意这种最高位有进位的情况
		{[]int{1, 2, 3}, []int{1, 2, 4}},
		{[]int{}, []int{}},
		{[]int{1, 2, 9}, []int{1, 3, 0}},
	}
	for _, tc := range tests {
		got := PlusOne(tc.input)
		if !reflect.DeepEqual(got, tc.want) {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestSingleNumber(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{4, 1, 2, 1, 2}, 4},
		{[]int{2, 2, 1}, 1},
	}
	for _, tc := range tests {
		got := SingleNumber(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestSingleNumber2(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{2, 2, 3, 2}, 3},
	}
	for _, tc := range tests {
		got := SingleNumber22(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestMajorityElement(t *testing.T) {
	type test struct {
		input []int
		want  int
	}
	tests := []test{
		{[]int{1}, 1},
		{[]int{3, 3, 1}, 3},
		{[]int{2, 2, 1, 1, 1, 2, 2}, 2},
	}
	for _, tc := range tests {
		got := MajorityElement3(tc.input)
		if got != tc.want {
			t.Log(cmp.Diff(tc.want, got))
		}
	}
}

func TestTwoSum(t *testing.T) {
	numbers := []int{3, 24, 50, 79, 88, 150, 345}
	target := 200
	t.Log(TwoSum5(numbers, target))
}

func TestCountPrimes(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{5}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := CountPrimes(tt.args.n); got != tt.want {
				t.Errorf("CountPrimes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBinarySearchFindLeftBound(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 2, 2, 4}, 2}, 1},
		{"right", args{[]int{1, 2, 2, 4}, 6}, -1},
		{"left", args{[]int{1, 2, 2, 4}, 0}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BinarySearchFindLeftBound(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("BinarySearchFindLeftBound() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBinarySearchFindRightBound(t *testing.T) {
	type args struct {
		nums   []int
		target int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 2, 2, 4}, 2}, 2},
		{"right", args{[]int{1, 2, 2, 4}, 6}, -1},
		{"left", args{[]int{1, 2, 2, 4}, 0}, -1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := BinarySearchFindRightBound(tt.args.nums, tt.args.target); got != tt.want {
				t.Errorf("BinarySearchFindRightBound() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestSubarraySum(t *testing.T) {
	type args struct {
		nums []int
		k    int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 1}, 2}, 2},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := SubarraySum3(tt.args.nums, tt.args.k); got != tt.want {
				t.Errorf("SubarraySum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMoveZeroes(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
	}{
		{"normal", args{[]int{0, 1, 0, 3, 12}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			MoveZeroes2(tt.args.nums)
		})
	}
}

func TestRemoveElement(t *testing.T) {
	type args struct {
		nums []int
		val  int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{3, 2, 2, 3}, 3}, 2},
		{"normal2", args{[]int{0, 1, 2, 2, 3, 0, 4, 2}, 2}, 5},
		{"normal2", args{[]int{1, 2, 3, 5, 4}, 5}, 4},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveElement(tt.args.nums, tt.args.val); got != tt.want {
				t.Errorf("RemoveElement() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoveDuplicates(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 2}}, 2},
		{"normal2", args{[]int{0, 0, 1, 1, 1, 2, 2, 3, 3, 4}}, 5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveDuplicates(tt.args.nums); got != tt.want {
				t.Errorf("RemoveDuplicates() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRemoveDuplicates2(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"normal", args{[]int{1, 1, 1, 2, 2, 3}}, 5},
		{"normal", args{[]int{0, 0, 1, 1, 1, 1, 2, 3, 3}}, 7},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := RemoveDuplicates2(tt.args.nums); got != tt.want {
				t.Errorf("RemoveDuplicates2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMerge(t *testing.T) {
	num1 := make([]int, 6)
	copy(num1, []int{4, 5, 6})
	m := 3
	num2 := []int{1, 2, 3}
	n := 3
	Merge(num1, m, num2, n)
	fmt.Println(num1)
}

func TestFindKthLargest(t *testing.T) {
	nums := []int{3, 2, 1, 5, 6, 4}
	k := 2
	// nums := []int{1}
	// k := 1

	t.Log(FindKthLargest3(nums, k))
}

func TestMaxSubArray(t *testing.T) {
	type args struct {
		nums []int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"n1", args{[]int{-2, 1, -3, 4, -1, 2, 1, -5, 4}}, 6},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := MaxSubArray3(tt.args.nums); got != tt.want {
				t.Errorf("MaxSubArray() = %v, want %v", got, tt.want)
			}
		})
	}
}

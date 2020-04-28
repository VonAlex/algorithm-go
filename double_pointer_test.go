package leetcode

import "testing"

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

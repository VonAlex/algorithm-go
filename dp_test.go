package leetcode

import "testing"

func TestFib(t *testing.T) {
	type args struct {
		n int
	}
	tests := []struct {
		name string
		args args
		want int
	}{
		{"0 test", args{0}, 0},
		{"n1", args{2}, 1},
		{"n2", args{5}, 5},
		{"n3", args{45}, 134903163},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Fib3(tt.args.n); got != tt.want {
				t.Errorf("Fib() = %v, want %v", got, tt.want)
			}
		})
	}
}

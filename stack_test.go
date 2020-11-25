package leetcode

import "testing"

func Test_isValid(t *testing.T) {
	type args struct {
		s string
	}
	tests := []struct {
		name string
		args args
		want bool
	}{
		{"case1", args{"()"}, true},
		{"case2", args{"()[]{}"}, true},
		{"case3", args{"(]"}, false},
		{"case4", args{"([)]"}, false},
		{"case5", args{"{[]}"}, true},
		{"case6", args{"[({(())}[()])]"}, true},
		{"case7", args{"["}, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isValid2(tt.args.s); got != tt.want {
				t.Errorf("isValid2() = %v, want %v", got, tt.want)
			}
		})
	}
}

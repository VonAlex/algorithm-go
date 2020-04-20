package leetcode

import (
	"reflect"
	"testing"
)

// 右子树失衡
var root = &TreeNode{
	Val: 1,
	Left: &TreeNode{
		Val: 2,
		Left: &TreeNode{
			Val: 4,
		},
		Right: &TreeNode{
			Val: 5,
		},
	},
	Right: &TreeNode{
		Val: 3,
	},
}

// 左子树失衡
var root2 = &TreeNode{
	Val: 1,
	Right: &TreeNode{
		Val: 2,
		Left: &TreeNode{
			Val: 4,
		},
		Right: &TreeNode{
			Val: 5,
		},
	},
	Left: &TreeNode{
		Val: 3,
	},
}

// 单节点树
var oneNodeRoot = &TreeNode{
	Val: 1,
}

func TestPreorderTraversal(t *testing.T) {
	type args struct {
		root *TreeNode
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"case one node", args{oneNodeRoot}, []int{1}},
		{"case left unbalance", args{root2}, []int{1, 3, 2, 4, 5}},
		{"case right unbalance", args{root}, []int{1, 2, 4, 5, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PreorderTraversal4(tt.args.root); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("PreorderTraversal() = %v, want %v", got, tt.want)
			}
		})
	}
}

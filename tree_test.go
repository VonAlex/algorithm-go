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

// 一颗二叉搜索树
var BSTroot = &TreeNode{
	Val: 5,
	Left: &TreeNode{
		Val: 2,
		Left: &TreeNode{
			Val: 1,
		},
		Right: &TreeNode{
			Val: 4,
			Left: &TreeNode{
				Val: 3,
			},
		},
	},
	Right: &TreeNode{
		Val: 6,
		Right: &TreeNode{
			Val: 7,
		},
	},
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
		{"case left unbalance", args{root2}, []int{3, 4, 5, 2, 1}},
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

func TestInorderTraversal(t *testing.T) {
	type args struct {
		root *TreeNode
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"case one node", args{oneNodeRoot}, []int{1}},
		{"case left unbalance", args{root2}, []int{3, 1, 4, 2, 5}},
		{"case right unbalance", args{root}, []int{4, 2, 5, 1, 3}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := InorderTraversal2(tt.args.root); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("InorderTraversal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestPostorderTraversal(t *testing.T) {
	type args struct {
		root *TreeNode
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{"case one node", args{oneNodeRoot}, []int{1}},
		{"case left unbalance", args{root2}, []int{3, 4, 5, 2, 1}},
		{"case right unbalance", args{root}, []int{4, 5, 2, 3, 1}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := PostorderTraversal2(tt.args.root); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("PostorderTraversal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestLevelOrderTraversal(t *testing.T) {
	type args struct {
		root *TreeNode
	}
	tests := []struct {
		name string
		args args
		want [][]int
	}{
		{"case one node", args{oneNodeRoot}, [][]int{{1}}},
		{"case left unbalance", args{root2}, [][]int{{1}, {3, 2}, {4, 5}}},
		{"case right unbalance", args{root}, [][]int{{1}, {2, 3}, {4, 5}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := LevelOrderTraversal2(tt.args.root); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("LevelOrderTraversal() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestBST(t *testing.T) {
	// t.Log(IsInBST(BSTroot, 8))
	// t.Log(InorderTraversal(InsertIntoBST2(BSTroot, 8)))
	// t.Log(InorderTraversal(DelFromBST(BSTroot, 2)))
	t.Log(isValidBST2(BSTroot))
}

func Test_sortedArrayToBST(t *testing.T) {
	nums := []int{-10, -3, 0, 5, 9}
	root := sortedArrayToBST(nums)
	t.Log(LevelOrderTraversal(root))
}

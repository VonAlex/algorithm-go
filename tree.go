package leetcode

import "container/list"

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

/**
 * LeetCode 题144 二叉树的前序遍历
 * https://leetcode-cn.com/problems/binary-tree-preorder-traversal/
 * 前序遍历：中 → 左 → 右
 */
// 解法 1：递归法
func PreorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, root.Val)
	res = append(res, PreorderTraversal(root.Left)...)
	res = append(res, PreorderTraversal(root.Right)...)
	return res
}

// 解法 2：迭代法1
// 使用了 container 包中的 list 来模拟栈
func PreorderTraversal2(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	stack := list.New()
	stack.PushBack(root)
	for stack.Len() > 0 {
		root = stack.Back().Value.(*TreeNode)
		res = append(res, root.Val)
		stack.Remove(stack.Back()) // 出栈
		if root.Right != nil {
			stack.PushBack(root.Right)
		}
		if root.Left != nil {
			stack.PushBack(root.Left)
		}
	}
	return res
}

// 解法 2：迭代法2
// 使用了 slice 来模拟栈，不引用其他包
func PreorderTraversal3(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	var stack []*TreeNode
	stack = append(stack, root)
	for len(stack) != 0 {
		index := len(stack) - 1
		root = stack[index] // 栈顶
		res = append(res, root.Val)

		stack = stack[:index] // 出栈
		if root.Right != nil {
			stack = append(stack, root.Right)
		}
		if root.Left != nil {
			stack = append(stack, root.Left)
		}
	}
	return res
}

// 解法 3：莫里斯遍历
// 每个前驱恰好访问两次，因此复杂度是 O(N)，其中 N 是顶点的个数，也就是树的大小。
// 我们在计算中不需要额外空间，但是输出需要包含 N 个元素，因此空间复杂度为 O(N)
func PreorderTraversal4(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	curr := root
	for curr != nil {
		if curr.Left == nil { // 左子树最深的节点
			res = append(res, curr.Val)
			curr = curr.Right
			continue
		}
		// 向左孩子走一步
		prev := curr.Left
		// 沿着右孩子一直向下访问，直到到达一个叶子节点（当前节点的中序遍历前驱节点）
		for prev.Right != nil && prev.Right != curr {
			prev = prev.Right
		}
		if prev.Right == nil {
			res = append(res, curr.Val)
			prev.Right = curr // 建立一条伪边到当前节点
			curr = curr.Left  // 访问左孩子
		} else { // 第二次访问到前驱节点
			prev.Right = nil  // right 重置为 nil
			curr = curr.Right // 移动到下一个顶点
		}
	}
	return res
}

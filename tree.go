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
	curr := root
	res = append(res, curr.Val)
	res = append(res, PreorderTraversal(curr.Left)...)
	res = append(res, PreorderTraversal(curr.Right)...)
	return res
}

// 解法 2：迭代法1
// 使用了 container 包中的 list 来模拟栈
func PreorderTraversal2(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	curr := root
	stack := list.New()
	stack.PushBack(curr)
	for stack.Len() > 0 {
		curr = stack.Back().Value.(*TreeNode)
		res = append(res, curr.Val)
		stack.Remove(stack.Back()) // 出栈
		if curr.Right != nil {
			stack.PushBack(curr.Right)
		}
		if curr.Left != nil {
			stack.PushBack(curr.Left)
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
	curr := root
	stack = append(stack, curr)
	for len(stack) != 0 {
		index := len(stack) - 1
		curr = stack[index] // 栈顶
		res = append(res, curr.Val)
		stack = stack[:index] // 出栈
		if curr.Right != nil {
			stack = append(stack, curr.Right) // 右孩子后遍历，先入栈
		}
		if curr.Left != nil {
			stack = append(stack, curr.Left)
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
			curr = curr.Right // 左孩子遍历完了，开始遍历右孩子
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

/**
 * LeetCode 题144 二叉树的中序遍历
 * https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
 * 中序遍历：左 → 中 → 右
 */
// 解法 1：递归法
func InorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	curr := root
	res = append(res, InorderTraversal(curr.Left)...)
	res = append(res, curr.Val)
	res = append(res, InorderTraversal(curr.Right)...)
	return res
}

// 解法 2：迭代法
func InorderTraversal2(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	var stack []*TreeNode
	curr := root
	for len(stack) > 0 || curr != nil {
		if curr != nil {
			stack = append(stack, curr)
			curr = curr.Left // 找到最深的左孩子
		} else {
			index := len(stack) - 1
			curr = stack[index] // 栈顶
			res = append(res, curr.Val)
			stack = stack[:index] // 出栈
			curr = curr.Right     // 最后遍历右孩子
		}
	}
	return res
}

// 解法 3：莫里斯遍历
// 与前序遍历的不同之处在于打印节点的顺序不同，其他是相同的
func InorderTraversal3(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	curr := root
	for curr != nil {
		if curr.Left == nil {
			res = append(res, curr.Val)
			curr = curr.Right
			continue
		}
		pre := curr.Left
		for pre.Right != nil && pre.Right != curr {
			pre = pre.Right
		}
		if pre.Right == nil {
			pre.Right = curr
			curr = curr.Left
		} else {
			pre.Right = nil
			res = append(res, curr.Val)
			curr = curr.Right
		}
	}
	return res
}

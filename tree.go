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
	curr := root
	stack := list.New()
	stack.PushBack(curr)
	for stack.Len() > 0 {
		curr = stack.Remove(stack.Back()).(*TreeNode) // 出栈
		res = append(res, curr.Val)
		if curr.Right != nil {
			stack.PushBack(curr.Right) // 先右
		}
		if curr.Left != nil {
			stack.PushBack(curr.Left) // 后左
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
// 在Morris方法中不需要为每个节点额外分配指针指向其前驱（predecessor）和后继节点（successor），
// 只需要利用叶子节点中的左右空指针指向某种顺序遍历下的前驱节点或后继节点就可以了。

// 每个前驱节点恰好访问两次，因此复杂度是 O(N)，其中 N 是顶点的个数，也就是树的大小。
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
			prev.Right = nil  // right 重置为 nil，取消伪边
			curr = curr.Right // 移动到右孩子
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
	res = append(res, InorderTraversal(root.Left)...)
	res = append(res, root.Val)
	res = append(res, InorderTraversal(root.Right)...)
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

			curr = curr.Right // 最后遍历右孩子
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

/**
 * LeetCode 题144 二叉树的后序遍历
 * https://leetcode-cn.com/problems/binary-tree-inorder-traversal/
 * 后序遍历：左 → 右 → 中
 */
// 解法 1：递归法
func PostorderTraversal(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	res = append(res, PostorderTraversal(root.Left)...)
	res = append(res, PostorderTraversal(root.Right)...)
	res = append(res, root.Val)
	return res
}

// 解法 2：迭代法
// 需要用到两个辅助栈
func PostorderTraversal2(root *TreeNode) []int {
	var res []int
	if root == nil {
		return res
	}
	// helpStack 中存放的是后序遍历的逆序节点
	var stack, helpStack []*TreeNode
	curr := root
	stack = append(stack, curr)
	for len(stack) > 0 {
		index := len(stack) - 1
		curr = stack[index]   // 栈顶
		stack = stack[:index] // 出栈

		helpStack = append(helpStack, curr)
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}
	}
	for i := len(helpStack) - 1; i >= 0; i-- {
		res = append(res, helpStack[i].Val)
	}
	return res
}

/**
 * LeetCode 题102 二叉树的层遍历
 * hhttps://leetcode-cn.com/problems/binary-tree-level-order-traversal/
 */
// 解法 1：递归法
// 深度遍历 DFS
func LevelOrderTraversal(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	var _dfs func(curr *TreeNode, level int)
	_dfs = func(curr *TreeNode, level int) {
		if curr == nil {
			return
		}
		if level == len(res) { // 新的一层
			res = append(res, []int{})
		}
		res[level] = append(res[level], curr.Val) // 相应的层添加节点
		_dfs(curr.Left, level+1)                  // 先左后右
		_dfs(curr.Right, level+1)
	}
	curr := root
	_dfs(curr, 0)
	return res
}

// 解法 2：迭代法
// 广度遍历 BFS
func LevelOrderTraversal2(root *TreeNode) [][]int {
	var res [][]int
	if root == nil {
		return res
	}
	queue := list.New()
	queue.PushFront(root) // 每次从头部插入新节点
	for queue.Len() > 0 {
		var currLevel []int
		currLevelNodeNums := queue.Len() // 当前层有多少个节点
		for i := 0; i < currLevelNodeNums; i++ {
			curr := queue.Remove(queue.Back()).(*TreeNode) // 每次从后面取出老节点
			currLevel = append(currLevel, curr.Val)
			if curr.Left != nil {
				queue.PushFront(curr.Left) // 先左孩子
			}
			if curr.Right != nil {
				queue.PushFront(curr.Right) // 后右孩子
			}
		}
		res = append(res, currLevel)
	}
	return res
}

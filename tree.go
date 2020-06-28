package leetcode

import (
	"container/list"
	"math"
)

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

// 二叉树框架
// func Traverse(root *TreeNode) {
// 	// root 需要做什么
// 	Traverse(root.Left)
// 	Traverse(root.Right)
// }

/*
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
		curr = stack[len(stack)-1]   // 栈顶
		stack = stack[:len(stack)-1] // 出栈

		res = append(res, curr.Val)
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
		for curr != nil {
			stack = append(stack, curr)
			curr = curr.Left // 找到最深的左孩子
		}
		curr = stack[len(stack)-1]   // 栈顶
		stack = stack[:len(stack)-1] // 出栈

		res = append(res, curr.Val)
		curr = curr.Right // 最后遍历右孩子
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

/*
 * LeetCode 题102 二叉树的层遍历
 * https://leetcode-cn.com/problems/binary-tree-level-order-traversal/
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

// 方法 1：递归法 DFS
// 空间复杂度为 O(n)，n 是树的高度
// 如果树很高，可能导致递归栈溢出
func InvertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	left := root.Left
	right := root.Right
	root.Left = InvertTree(right)
	root.Right = InvertTree(left)
	return root
}

// 方法 2：迭代1 后序遍历
func InvertTree2(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	curr := root
	stack := []*TreeNode{}
	stack = append(stack, curr)
	for len(stack) != 0 {
		curr = stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		curr.Left, curr.Right = curr.Right, curr.Left
		if curr.Left != nil {
			stack = append(stack, curr.Left)
		}
		if curr.Right != nil {
			stack = append(stack, curr.Right)
		}
	}
	return root
}

// 方法 2：迭代2 层序遍历 BFS
func InvertTree3(root *TreeNode) *TreeNode {
	if root == nil {
		return root
	}
	curr := root
	queue := []*TreeNode{}
	queue = append(queue, curr)
	for len(queue) != 0 {
		curr = queue[0]
		queue = queue[1:]
		curr.Left, curr.Right = curr.Right, curr.Left
		if curr.Left != nil {
			queue = append(queue, curr.Left)
		}
		if curr.Right != nil {
			queue = append(queue, curr.Right)
		}
	}
	return root
}

// 判断两棵二叉树是否完全相同
func IsSameTree(root1, root2 *TreeNode) bool {
	// 都为空
	if root1 == nil && root2 == nil {
		return true
	}
	// 只有一个为空
	if root1 == nil || root2 == nil {
		return false
	}
	// 都不为空，但是 val 不等
	if root1.Val != root2.Val {
		return false
	}
	// 比较完本节点，比较左右子节点
	return IsSameTree(root1.Left, root2.Left) &&
		IsSameTree(root1.Right, root2.Right)
}

/*
 * LeetCode T98. 验证二叉搜索树
 * https://leetcode-cn.com/problems/validate-binary-search-tree/
 * 给定一个二叉树，判断其是否是一个有效的二叉搜索树。
 *
 * 假设一个二叉搜索树具有如下特征：
 * 		节点的左子树只包含小于当前节点的数。
 * 		节点的右子树只包含大于当前节点的数。
 * 		所有左子树和右子树自身必须也是二叉搜索树。
 */
// 二叉搜索树(Binary Search Tree，简称 BST)

// 方法 1：递归法
// 时间复杂度和空间复杂度都是 O(n)， n 为节点个数
// 在递归调用的时候二叉树的每个节点最多被访问一次，因此时间复杂度为 O(n)。
// 递归函数在递归过程中需要为每一层递归函数分配栈空间，所以这里需要额外的空间且该空间取决于递归的深度，即二叉树的高度。
// 最坏情况下二叉树为一条链，树的高度为 n ，递归最深达到 n 层，故最坏情况下空间复杂度为 O(n)
func IsValidBST(root *TreeNode) bool {
	var _helper func(*TreeNode, int, int) bool
	_helper = func(t *TreeNode, max, min int) bool {
		if t == nil {
			return true
		}
		if t.Val >= max || t.Val <= min {
			return false
		}
		// 左子树上所有节点的值均小于它的根节点的值
		if !_helper(t.Left, t.Val, min) {
			return false
		}
		// 右子树上所有节点的值均大于它的根节点的值
		if !_helper(t.Right, max, t.Val) {
			return false
		}
		return true
	}
	curr := root
	return _helper(curr, math.MaxInt64, math.MinInt64)

}

// 方法 2：中序遍历
// BST 的中序遍历是一个升序数组，如果遍历的时候发现，当前的 node 值<= 前一个，那就表示这不是一个 BST
func isValidBST2(root *TreeNode) bool {
	if root == nil {
		return true
	}
	var stack []*TreeNode
	prev := math.MinInt64 // 起始值

	curr := root
	for len(stack) != 0 || curr != nil {
		for curr != nil {
			stack = append(stack, curr)
			curr = curr.Left
		}

		curr = stack[len(stack)-1]
		stack = stack[:len(stack)-1]

		if curr.Val <= prev {
			return false
		}
		prev = curr.Val
		curr = curr.Right
	}
	return true
}

/*
 * LeetCode T700. 二叉搜索树中的搜索
 * https://leetcode-cn.com/problems/search-in-a-binary-search-tree/
 * 给定二叉搜索树（BST）的根节点和一个值。 你需要在BST中找到节点值等于给定值的节点。
 * 返回以该节点为根的子树。 如果节点不存在，则返回 NULL。
 */
// 方法 1：递归法
func SearchBST(root *TreeNode, target int) *TreeNode {
	if root == nil {
		return root
	}
	if root.Val == target {
		return root
	}
	if root.Val < target { // 右子树
		return SearchBST(root.Right, target)
	}
	return SearchBST(root.Left, target) // 左子树
}

// 方法 2：迭代法
func SearchBST2(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return root
	}
	curr := root
	for curr != nil {
		if curr.Val == val {
			return curr
		}
		if curr.Val < val {
			curr = curr.Right
		} else {
			curr = curr.Left
		}
	}
	return nil
}

/*
 * LeetCode T701. 二叉搜索树中的插入操作
 * https://leetcode-cn.com/problems/insert-into-a-binary-search-tree/
 * 给定二叉搜索树（BST）的根节点和要插入树中的值，将值插入二叉搜索树。
 * 返回插入后二叉搜索树的根节点。
 * 保证原始二叉搜索树中不存在新值。
 */
// 方法 1：递归法
// 递归的深度取决于 BST 的深度，最坏的时间复杂度为 n（n=节点数量），此时 BST 退化成一个链表
func InsertIntoBST(root *TreeNode, val int) *TreeNode {
	if root == nil { // 插入位置
		return &TreeNode{
			Val: val,
		}
	}
	if root.Val < val {
		root.Right = InsertIntoBST(root.Right, val)
	}
	if root.Val > val {
		root.Left = InsertIntoBST(root.Left, val)
	}
	return root // 返回节点
}

// 方法 2：迭代法
func InsertIntoBST2(root *TreeNode, val int) *TreeNode {
	if root == nil {
		return &TreeNode{
			Val: val,
		}
	}
	curr := root
	var parent *TreeNode

	// 找到插入位置的父节点
	for curr != nil {
		parent = curr
		if curr.Val < val {
			curr = curr.Right
		} else {
			curr = curr.Left
		}
	}
	newNode := &TreeNode{
		Val: val,
	}
	if parent.Val < val {
		parent.Right = newNode
	} else {
		parent.Left = newNode
	}
	return root
}

/*
 * LeetCode T450. 删除二叉搜索树中的节点
 * https://leetcode-cn.com/problems/delete-node-in-a-bst/
 * 给定一个二叉搜索树的根节点 root 和一个值 key，删除二叉搜索树中的 key 对应的节点，
 * 并保证二叉搜索树的性质不变。返回二叉搜索树（有可能被更新）的根节点的引用。
 *
 * 一般来说，删除节点可分为两个步骤：
 * 首先找到需要删除的节点；
 * 如果找到了，删除它。
 * 说明： 要求算法时间复杂度为 O(h)，h 为树的高度。
 */
func DelFromBST(root *TreeNode, target int) *TreeNode {
	if root == nil {
		return nil
	}
	// 三种情况处理
	if root.Val == target {
		if root.Left == nil {
			return root.Right
		}
		if root.Right == nil {
			return root.Left
		}
		// 找到右子树的最小值
		minNode := getBSTMinNode(root.Right)
		// 与当前节点交换，因为要删掉当前节点，又要不破坏 BST 的性质，因此需要找到右子树的最小值
		root.Val, minNode.Val = minNode.Val, root.Val
		// 在右子树中删掉 target
		root.Right = DelFromBST(root.Right, target)
	} else if root.Val < target {
		root.Right = DelFromBST(root.Right, target)
	} else {
		root.Left = DelFromBST(root.Left, target)
	}
	return root
}

// BST 中最小的值就是最左节点，也就是中序遍历的第一个值
func getBSTMinNode(node *TreeNode) *TreeNode {
	if node == nil {
		return nil
	}
	for node.Left != nil {
		node = node.Left
	}
	return node
}

/*
 * LeetCode 面试题54. 二叉搜索树的第k大节点
 * 给定一棵二叉搜索树，请找出其中第k大的节点
 */
// 根据 BTS 的性质，中序遍历是一个递增数组，所以中序遍历的逆序就是一个递减数组（右 → 中 → 左）
// 方法 1：迭代法
func KthLargest(root *TreeNode, k int) int {
	if root == nil {
		return -1
	}
	curr := root
	stack := []*TreeNode{}
	for len(stack) != 0 || curr != nil {
		for curr != nil { // 先找到最右节点
			stack = append(stack, curr)
			curr = curr.Right
		}
		curr = stack[len(stack)-1] // 再遍历本节点
		stack = stack[:len(stack)-1]
		k--
		if k == 0 {
			return curr.Val
		}
		curr = curr.Left // 再找到左孩子
	}
	return -1 // 考虑不符合条件的返回值
}

// 方法 2：递归法
func KthLargest2(root *TreeNode, k int) int {
	var result int
	var healper func(*TreeNode)
	healper = func(node *TreeNode) {
		if node == nil {
			return
		}
		healper(node.Right)
		k--
		if k == 0 {
			result = node.Val
			return
		}
		healper(node.Left)
	}
	healper(root)
	return result
}

/*
 * LeetCode T105. 从前序与中序遍历序列构造二叉树
 * https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
 *
 * 根据一棵树的前序遍历与中序遍历构造二叉树。
 * 注意: 你可以假设树中没有重复的元素。
 *
 * 例如，给出
 * 前序遍历 preorder = [3,9,20,15,7]
 * 中序遍历 inorder = [9,3,15,20,7]
 * 返回如下的二叉树：
 *     3
 *    / \
 *   9  20
 *     /  \
 *    15   7
 */
// 前序遍历的形式总是 [ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ], 即根节点总是前序遍历中的第一个节点。
// 而中序遍历的形式总是 [ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]
// 只要我们在中序遍历中定位到根节点，那么我们就可以分别知道左子树和右子树中的节点数目
// 参考 https://mp.weixin.qq.com/s/QHt9fGP-q8RAs8GI7fP3hw
func buildTree(preorder []int, inorder []int) *TreeNode {
	preorderLen := len(preorder)
	inorderLen := len(inorder)
	if preorderLen == 0 || inorderLen == 0 {
		return nil
	}
	if preorderLen != inorderLen {
		return nil
	}
	// 使用 hash 表辅助定位根节点在 inorder 数组中的 idx
	// 空间复杂度 O(n)
	valsInorder := make(map[int]int)
	for i, val := range inorder {
		valsInorder[val] = i
	}

	var _build func(int, int, int) *TreeNode
	// 递归构建左右子树
	// 空间复杂度 O(h), h 为树的高度
	// preRootIdx 表示 root 节点在 preorder 数组中的 idx
	// inLeftIdx 表示中序遍历左边界
	// inRightIdx 表示中序遍历右边界
	_build = func(preRootIdx, inLeftIdx, inRightIdx int) *TreeNode {
		if inLeftIdx > inRightIdx {
			return nil
		}
		root := &TreeNode{
			Val: preorder[preRootIdx],
		}
		inRootIdx := valsInorder[root.Val]
		// 在 inorder 的 [inLeftIdx,inRootIdx-1] 范围构建左子树，在 preorder 的根节点索引是 preRootIdx+1
		root.Left = _build(preRootIdx+1, inLeftIdx, inRootIdx-1)
		// 在 inorder 的 [inRootIdx+1, inRightIdx] 范围构建右子树
		root.Right = _build(preRootIdx+(inRootIdx-inLeftIdx)+1, inRootIdx+1, inRightIdx)
		return root
	}
	return _build(0, 0, inorderLen-1)
}

/*
 * LeetCode T104. 二叉树的最大深度
 * https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/
 * 给定一个二叉树，找出其最大深度。
 * 二叉树的深度为根节点到最远叶子节点的最长路径上的节点数。
 * 说明: 叶子节点是指没有子节点的节点。
 */
// 方法 1：递归 DFS
// 时间复杂度 O(n), 空间复杂度最差 O(n)，O(logn)
func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}
	lh := maxDepth(root.Left)
	rh := maxDepth(root.Right)
	return maxInt(lh, rh) + 1
}

func maxInt(x, y int) int {
	if x > y {
		return x
	}
	return y
}

// 方法 2：迭代 BFS
// 时/空间复杂度 O(n)，实际上就是层序遍历
func maxDepth2(root *TreeNode) int {
	if root == nil {
		return 0
	}
	maxDepth := 0
	curr := root
	q := list.New()
	q.PushBack(curr)
	for q.Len() > 0 {
		maxDepth++
		size := q.Len()
		for i := 0; i < size; i++ {
			// 队列先进先出
			c := q.Remove(q.Front()).(*TreeNode)
			if c.Left != nil {
				q.PushBack(c.Left)
			}
			if c.Right != nil {
				q.PushBack(c.Right)
			}
		}
	}
	return maxDepth
}

/* LeetCode T226. 翻转二叉树
 * https://leetcode-cn.com/problems/invert-binary-tree/
 * 剑指 Offer 27. 二叉树的镜像
 * https://leetcode-cn.com/problems/er-cha-shu-de-jing-xiang-lcof/
 * 请完成一个函数，输入一个二叉树，该函数输出它的镜像。(翻转一棵二叉树)
 */
// 思路：先前序遍历这棵树，如果遍历到的节点有子节点，那么就交换它的两个子节点。
// 当交换完所有非叶子节点的左右子节点后，就可以得到树的镜像
func mirrorTree(root *TreeNode) *TreeNode {
	var _mirrorTreeRecur func(*TreeNode)
	_mirrorTreeRecur = func(t *TreeNode) {
		if t == nil {
			return
		}
		if t.Left == nil && t.Right == nil {
			return
		}
		t.Left, t.Right = t.Right, t.Left
		if t.Left != nil {
			_mirrorTreeRecur(t.Left)
		}
		if t.Right != nil {
			_mirrorTreeRecur(t.Right)
		}
		return
	}
	curr := root
	_mirrorTreeRecur(curr)
	return root
}

/*
 * LeetCode T101. 对称二叉树
 * https://leetcode-cn.com/problems/symmetric-tree/
 * 剑指 Offer 28. 对称的二叉树
 * https://leetcode-cn.com/problems/dui-cheng-de-er-cha-shu-lcof/
 * 给定一个二叉树，检查它是否是镜像对称的。
 * (请实现一个函数，用来判断一棵二叉树是不是对称的。如果一棵二叉树和它的镜像一样，那么它是对称的。)
 */
// 方法 1：递归法
// 时/空间复杂度 O(n)
func isSymmetric(root *TreeNode) bool {
	if root == nil {
		return true
	}
	curr := root
	var _ismirror func(*TreeNode, *TreeNode) bool
	_ismirror = func(curr1, curr2 *TreeNode) bool {
		if curr1 == nil && curr2 == nil {
			return true
		}
		if curr1 == nil || curr2 == nil {
			return false
		}
		if curr1.Val != curr2.Val {
			return false
		}
		return _ismirror(curr1.Left, curr2.Right) && _ismirror(curr1.Right, curr2.Left)
	}
	return _ismirror(curr, curr)
}

// 方法 2：迭代法
// 需要借助队列做层遍历，也可以使用数组代替
// 时/空间复杂度 O(n)
func isSymmetric2(root *TreeNode) bool {
	if root == nil {
		return true
	}
	q := list.New()
	curr := root
	q.PushBack(curr)
	q.PushBack(curr)
	for q.Len() > 0 {
		t1 := q.Remove(q.Front()).(*TreeNode)
		t2 := q.Remove(q.Front()).(*TreeNode)
		if t1 == nil && t2 == nil {
			continue
		}
		if t1 == nil || t2 == nil {
			return false
		}
		if t1.Val != t2.Val {
			return false
		}
		q.PushBack(t1.Left)
		q.PushBack(t2.Right)
		q.PushBack(t1.Right)
		q.PushBack(t2.Left)
	}
	return true
}

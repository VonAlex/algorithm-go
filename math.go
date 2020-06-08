package leetcode

/*
 * LeetCode T202. 快乐数
 * https://leetcode-cn.com/problems/happy-number/
 *
 * 编写一个算法来判断一个数 n 是不是快乐数。
 * 「快乐数」定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，
 * 然后重复这个过程直到这个数变为 1，也可能是 无限循环 但始终变不到 1。如果 可以变为  1，
 * 那么这个数就是快乐数。如果 n 是快乐数就返回 True ；不是，则返回 False 。
 * 示例:
 * 		输入：19
 * 		输出：true
 *		解释：
 *      1^2 + 9^2 = 82
 *      8^2 + 2^2 = 68
 *		6^2 + 8^2 = 100
 *		1^2 + 0^2 + 0^2 = 1
 */
// 意识到我们实际有个链表，那么这个问题就可以转换为检测一个链表是否有环。
// 时间复杂度：O(logn)
func isHappy(n int) bool {
	_step := func(n int) int {
		sum := 0
		for n != 0 {
			sum += (n % 10) * (n % 10)
			n /= 10
		}
		return sum
	}

	slow := n
	fast := _step(n)
	// fast == 1 说明是快乐数
	// fast == slow 说明不是快乐数
	for fast != 1 && fast != slow {
		slow = _step(slow)
		fast = _step(_step(fast))
	}
	return fast == 1
}

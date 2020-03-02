package leetcode

import "container/list"

/**
 * LeetCode 题 225 用队列实现栈
 * https://leetcode-cn.com/problems/implement-stack-using-queues/
 *
 * 使用队列实现栈的下列操作：
 * 		push(x) -- 元素 x 入栈
 * 		pop() -- 移除栈顶元素
 * 		top() -- 获取栈顶元素
 * 		empty() -- 返回栈是否为空
 */

type MyStack struct {
	*list.List
}

/** Initialize your data structure here. */
func Constructor() MyStack {
	stack := MyStack{
		list.New(),
	}
	return stack
}

/** Push element x onto stack. */
func (this *MyStack) Push(x int) {
	this.PushBack(x)
	return
}

/** Removes the element on top of the stack and returns that element. */
func (this *MyStack) Pop() int {
	top := this.Back()
	this.Remove(top)
	return top.Value.(int)
}

/** Get the top element. */
func (this *MyStack) Top() int {
	return this.Back().Value.(int)
}

/** Returns whether the stack is empty. */
func (this *MyStack) Empty() bool {
	return this.Len() == 0
}

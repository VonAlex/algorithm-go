package leetcode

/**
 * LeetCode T146. LRU缓存机制
 * https://leetcode-cn.com/problems/lru-cache/
 *
 * 运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制。
 * 它应该支持以下操作： 获取数据 get 和 写入数据 put 。
 */

// 参考题解 1：https://mp.weixin.qq.com/s/Q8Mg_EhDvPVRIaDv6m0Nlg
// 参考题解 2：https://leetcode-cn.com/problems/lru-cache/solution/lru-ce-lue-xiang-jie-he-shi-xian-by-labuladong/

// 方法：hash 表 + 双链表
// cache 这个数据结构必要的条件：查找快，插入快，删除快，有顺序之分。
// 在 O(1) 的时间找到节点，所以使用 hash 表
// 支持节点快速的移动、增删，所以使用双链表

// 当缓存容量已满，我们不仅仅要删除最后一个 Node 节点，还要把 map 中映射到该节点的 key 同时删除，
// 而这个 key 只能由 Node 得到。如果 Node 结构中只存储 val，那么我们就无法得知 key 是什么，就无法删除 map 中的键，造成错误。
// 这就是“为什么要在链表中同时存储 key 和 val，而不是只存储 val”
type Node struct {
	key  int
	val  int
	prev *Node
	next *Node
}

type LRUCache struct {
	cache map[int]*Node
	// 在双向链表的实现中，使用一个伪头部（dummy head）和伪尾部（dummy tail）标记界限，
	// 这样在添加节点和删除节点的时候就不需要检查相邻的节点是否存在。
	head     *Node
	tail     *Node
	capacity int // 容量
	size     int // 目前有多少元素
}

func NewLRUCache(capacity int) LRUCache {
	l := LRUCache{
		make(map[int]*Node),
		&Node{},
		&Node{},
		capacity,
		0,
	}
	l.head.next = l.tail
	l.tail.prev = l.head
	return l
}

func (l *LRUCache) Get(key int) int {
	node, ok := l.cache[key]
	if !ok {
		return -1
	}
	// 更新访问到的节点到头部
	l.moveToHead(node)
	return node.val
}

func (l *LRUCache) Put(key int, value int) {
	node, ok := l.cache[key]
	if ok { // 如果能找到 key，把它移动到头部
		node.val = value
		l.moveToHead(node)
		return
	}
	// 否则，新建一个节点
	// cache 中有空位置时，可以直接加到头部
	// 否则，从尾部踢掉最老的节点
	node = &Node{
		key: key,
		val: value,
	}
	l.addToHead(node)
	if l.size > l.capacity {
		l.removeTail()
		return
	}
}

func (l *LRUCache) remove(node *Node) {
	node.prev.next = node.next
	node.next.prev = node.prev

	delete(l.cache, node.key)
	l.size--
}

// cache 满了以后，从尾部删除
func (l *LRUCache) removeTail() *Node {
	tailNode := l.tail.prev
	l.remove(tailNode)
	return tailNode
}

// 新加或者访问到的元素放到头部
func (l *LRUCache) moveToHead(node *Node) {
	l.remove(node)    // 先删掉
	l.addToHead(node) // 再加到头部
}

func (l *LRUCache) addToHead(node *Node) {
	node.prev = l.head
	node.next = l.head.next
	l.head.next.prev = node
	l.head.next = node

	l.cache[node.key] = node
	l.size++
}

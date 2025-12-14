// Vue组件定义

// 工作台视图
const DeskView = {
    template: `
        <div>
            <h2 class="text-2xl font-bold mb-6">工作台</h2>

            <!-- 统计卡片 -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">总资产</h3>
                    <p class="text-2xl font-bold text-green-600">${{ formatNumber(balance.total || 0) }}</p>
                    <p class="text-sm text-gray-500 mt-2">USDT</p>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">总盈亏</h3>
                    <p :class="`text-2xl font-bold ${totalPnl >= 0 ? 'text-green-600' : 'text-red-600'}`">
                        ${{ formatNumber(totalPnl) }}
                    </p>
                    <p class="text-sm text-gray-500 mt-2">今日</p>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">持仓数</h3>
                    <p class="text-2xl font-bold text-blue-600">{{ positionCount }}</p>
                    <p class="text-sm text-gray-500 mt-2">个交易对</p>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">今日决策</h3>
                    <p class="text-2xl font-bold text-purple-600">{{ decisionCount }}</p>
                    <p class="text-sm text-gray-500 mt-2">个决策</p>
                </div>
            </div>

            <!-- 实时持仓 -->
            <div class="bg-white rounded-lg shadow mb-8">
                <div class="px-6 py-4 border-b">
                    <h3 class="text-lg font-semibold">实时持仓</h3>
                </div>
                <div class="p-6">
                    <table class="w-full" v-if="positions.length">
                        <thead>
                            <tr class="text-left text-gray-600">
                                <th class="pb-3">交易对</th>
                                <th class="pb-3">方向</th>
                                <th class="pb-3">数量</th>
                                <th class="pb-3">开仓价</th>
                                <th class="pb-3">当前价</th>
                                <th class="pb-3">盈亏</th>
                                <th class="pb-3">操作</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="pos in positions" :key="pos.symbol" class="border-t">
                                <td class="py-3">{{ pos.symbol }}</td>
                                <td class="py-3">
                                    <span :class="pos.side === 'long' ? 'text-green-600' : 'text-red-600'">
                                        {{ pos.side === 'long' ? '多' : '空' }}
                                    </span>
                                </td>
                                <td class="py-3">{{ formatNumber(pos.size) }}</td>
                                <td class="py-3">${{ formatNumber(pos.entry_price) }}</td>
                                <td class="py-3">${{ formatNumber(pos.current_price) }}</td>
                                <td class="py-3">
                                    <span :class="(pos.unrealized_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'">
                                        ${{ formatNumber(pos.unrealized_pnl) }}
                                    </span>
                                </td>
                                <td class="py-3">
                                    <router-link :to="'/positions/' + pos.symbol"
                                                  class="text-blue-600 hover:text-blue-800">
                                        详情
                                    </router-link>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    <p v-else class="text-gray-500">暂无持仓</p>
                </div>
            </div>

            <!-- 最近决策 -->
            <div class="bg-white rounded-lg shadow">
                <div class="px-6 py-4 border-b flex justify-between items-center">
                    <h3 class="text-lg font-semibold">最近决策</h3>
                    <router-link to="/decisions" class="text-blue-600 hover:text-blue-800">查看全部</router-link>
                </div>
                <div class="p-6">
                    <table class="w-full" v-if="decisions.length">
                        <thead>
                            <tr class="text-left text-gray-600">
                                <th class="pb-3">时间</th>
                                <th class="pb-3">交易对</th>
                                <th class="pb-3">操作</th>
                                <th class="pb-3">置信度</th>
                                <th class="pb-3">状态</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="dec in decisions" :key="dec.id" class="border-t">
                                <td class="py-3">{{ formatTime(dec.timestamp) }}</td>
                                <td class="py-3">{{ dec.symbol }}</td>
                                <td class="py-3">{{ formatAction(dec.action) }}</td>
                                <td class="py-3">{{ dec.confidence }}%</td>
                                <td class="py-3">
                                    <span :class="getStatusClass(dec.status)">
                                        {{ getStatusText(dec.status) }}
                                    </span>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                    <p v-else class="text-gray-500">暂无决策</p>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            balance: {},
            positions: [],
            decisions: [],
            totalPnl: 0,
            positionCount: 0,
            decisionCount: 0
        }
    },
    methods: {
        formatAction(action) {
            const map = {
                'enter_long': '开多',
                'enter_short': '开空',
                'close_long': '平多',
                'close_short': '平空',
                'hold': '持有'
            }
            return map[action] || action
        },
        getStatusClass(status) {
            const classes = {
                'pending': 'text-yellow-600',
                'executed': 'text-green-600',
                'failed': 'text-red-600',
                'cancelled': 'text-gray-600'
            }
            return classes[status] || ''
        },
        getStatusText(status) {
            const map = {
                'pending': '待执行',
                'executed': '已执行',
                'failed': '执行失败',
                'cancelled': '已取消'
            }
            return map[status] || status
        },
        async loadData() {
            try {
                // 加载余额
                const balanceRes = await axios.get('/api/live/freqtrade/balance')
                this.balance = balanceRes.data[0] || {}

                // 加载持仓
                const posRes = await axios.get('/api/live/freqtrade/positions')
                this.positions = posRes.data
                this.positionCount = this.positions.length
                this.totalPnl = this.positions.reduce((sum, pos) =>
                    sum + (pos.unrealized_pnl || 0) + (pos.realized_pnl || 0), 0)

                // 加载最近决策
                const decRes = await axios.get('/api/live/decisions', {
                    params: { limit: 10 }
                })
                this.decisions = decRes.data.decisions
                this.decisionCount = decRes.data.total
            } catch (error) {
                console.error('加载工作台数据失败:', error)
            }
        }
    },
    mounted() {
        this.loadData()

        // 监听持仓更新
        this.$root.$on('position-update', () => {
            this.loadData()
        })

        // 定期刷新
        setInterval(() => this.loadData(), 30000)
    }
}

// 决策列表视图
const DecisionsView = {
    template: `
        <div>
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">蓝皮书 - 决策历史</h2>
                <div class="flex space-x-4">
                    <select v-model="filters.symbol" @change="loadDecisions"
                            class="px-4 py-2 border rounded">
                        <option value="">所有交易对</option>
                        <option v-for="sym in symbols" :key="sym" :value="sym">{{ sym }}</option>
                    </select>
                    <select v-model="filters.action" @change="loadDecisions"
                            class="px-4 py-2 border rounded">
                        <option value="">所有操作</option>
                        <option value="enter_long">开多</option>
                        <option value="enter_short">开空</option>
                        <option value="close_long">平多</option>
                        <option value="close_short">平空</option>
                    </select>
                </div>
            </div>

            <div class="bg-white rounded-lg shadow">
                <table class="w-full">
                    <thead>
                        <tr class="text-left text-gray-600 border-b">
                            <th class="px-6 py-4">时间</th>
                            <th class="px-6 py-4">交易对</th>
                            <th class="px-6 py-4">操作</th>
                            <th class="px-6 py-4">置信度</th>
                            <th class="px-6 py-4">状态</th>
                            <th class="px-6 py-4">操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="dec in decisions" :key="dec.id" class="border-b hover:bg-gray-50">
                            <td class="px-6 py-4">{{ formatTime(dec.timestamp) }}</td>
                            <td class="px-6 py-4">{{ dec.symbol }}</td>
                            <td class="px-6 py-4">{{ formatAction(dec.action) }}</td>
                            <td class="px-6 py-4">
                                <div class="flex items-center">
                                    <span class="mr-2">{{ dec.confidence }}%</span>
                                    <div class="w-20 bg-gray-200 rounded-full h-2">
                                        <div class="bg-blue-600 h-2 rounded-full"
                                             :style="{width: dec.confidence + '%'}"></div>
                                    </div>
                                </div>
                            </td>
                            <td class="px-6 py-4">
                                <span :class="getStatusClass(dec.status)">
                                    {{ getStatusText(dec.status) }}
                                </span>
                            </td>
                            <td class="px-6 py-4">
                                <router-link :to="'/decisions/' + dec.id"
                                              class="text-blue-600 hover:text-blue-800">
                                    详情
                                </router-link>
                            </td>
                        </tr>
                    </tbody>
                </table>

                <!-- 分页 -->
                <div class="px-6 py-4 flex justify-between items-center border-t">
                    <div class="text-gray-600">
                        显示 {{ (page - 1) * pageSize + 1 }} - {{ Math.min(page * pageSize, total) }}
                        共 {{ total }} 条
                    </div>
                    <div class="flex space-x-2">
                        <button @click="page--" :disabled="page <= 1"
                                class="px-4 py-2 border rounded hover:bg-gray-100 disabled:opacity-50">
                            上一页
                        </button>
                        <span class="px-4 py-2">{{ page }} / {{ totalPages }}</span>
                        <button @click="page++" :disabled="page >= totalPages"
                                class="px-4 py-2 border rounded hover:bg-gray-100 disabled:opacity-50">
                            下一页
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            decisions: [],
            total: 0,
            page: 1,
            pageSize: 20,
            filters: {
                symbol: '',
                action: '',
                status: ''
            },
            symbols: []
        }
    },
    computed: {
        totalPages() {
            return Math.ceil(this.total / this.pageSize)
        }
    },
    methods: {
        formatAction(action) {
            const map = {
                'enter_long': '开多',
                'enter_short': '开空',
                'close_long': '平多',
                'close_short': '平空',
                'hold': '持有'
            }
            return map[action] || action
        },
        getStatusClass(status) {
            const classes = {
                'pending': 'text-yellow-600',
                'executed': 'text-green-600',
                'failed': 'text-red-600',
                'cancelled': 'text-gray-600'
            }
            return classes[status] || ''
        },
        getStatusText(status) {
            const map = {
                'pending': '待执行',
                'executed': '已执行',
                'failed': '执行失败',
                'cancelled': '已取消'
            }
            return map[status] || status
        },
        async loadDecisions() {
            try {
                const params = {
                    page: this.page,
                    page_size: this.pageSize,
                    ...this.filters
                }

                const res = await axios.get('/api/live/decisions', { params })
                this.decisions = res.data.decisions
                this.total = res.data.total

                // 提取交易对列表
                this.symbols = [...new Set(this.decisions.map(d => d.symbol))]
            } catch (error) {
                console.error('加载决策失败:', error)
            }
        }
    },
    watch: {
        page() {
            this.loadDecisions()
        }
    },
    mounted() {
        this.loadDecisions()
    }
}

// 持仓列表视图
const PositionsView = {
    template: `
        <div>
            <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold">红皮书 - 持仓管理</h2>
                <button @click="loadPositions"
                        class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                    刷新
                </button>
            </div>

            <!-- 统计信息 -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">总持仓</h3>
                    <p class="text-2xl font-bold">{{ stats.total || 0 }}</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">总敞口</h3>
                    <p class="text-2xl font-bold">${{ formatNumber(stats.total_exposure || 0) }}</p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">未实现盈亏</h3>
                    <p :class="`text-2xl font-bold ${stats.total_unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`">
                        ${{ formatNumber(stats.total_unrealized_pnl || 0) }}
                    </p>
                </div>
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm">已实现盈亏</h3>
                    <p :class="`text-2xl font-bold ${stats.total_realized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`">
                        ${{ formatNumber(stats.total_realized_pnl || 0) }}
                    </p>
                </div>
            </div>

            <!-- 持仓列表 -->
            <div class="bg-white rounded-lg shadow">
                <table class="w-full">
                    <thead>
                        <tr class="text-left text-gray-600 border-b">
                            <th class="px-6 py-4">交易对</th>
                            <th class="px-6 py-4">方向</th>
                            <th class="px-6 py-4">数量</th>
                            <th class="px-6 py-4">开仓价</th>
                            <th class="px-6 py-4">当前价</th>
                            <th class="px-6 py-4">盈亏</th>
                            <th class="px-6 py-4">盈亏率</th>
                            <th class="px-6 py-4">操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr v-for="pos in positions" :key="pos.symbol" class="border-b hover:bg-gray-50">
                            <td class="px-6 py-4 font-medium">{{ pos.symbol }}</td>
                            <td class="px-6 py-4">
                                <span :class="pos.side === 'long' ? 'text-green-600' : 'text-red-600'">
                                    {{ pos.side === 'long' ? '多' : '空' }}
                                </span>
                            </td>
                            <td class="px-6 py-4">{{ formatNumber(pos.size) }}</td>
                            <td class="px-6 py-4">${{ formatNumber(pos.entry_price) }}</td>
                            <td class="px-6 py-4">${{ formatNumber(pos.current_price || pos.entry_price) }}</td>
                            <td class="px-6 py-4">
                                <span :class="getPnlClass(pos)">
                                    ${{ formatNumber((pos.unrealized_pnl || 0) + (pos.realized_pnl || 0)) }}
                                </span>
                            </td>
                            <td class="px-6 py-4">
                                <span :class="getPnlClass(pos)">
                                    {{ formatPercent(calculatePnlPercent(pos)) }}
                                </span>
                            </td>
                            <td class="px-6 py-4">
                                <router-link :to="'/positions/' + pos.symbol"
                                              class="text-blue-600 hover:text-blue-800 mr-4">
                                    详情
                                </router-link>
                                <button @click="closePosition(pos)"
                                        class="text-red-600 hover:text-red-800">
                                    平仓
                                </button>
                            </td>
                        </tr>
                    </tbody>
                </table>
                <p v-if="!positions.length" class="text-center py-8 text-gray-500">
                    暂无持仓
                </p>
            </div>
        </div>
    `,
    data() {
        return {
            positions: [],
            stats: {}
        }
    },
    methods: {
        getPnlClass(pos) {
            const pnl = (pos.unrealized_pnl || 0) + (pos.realized_pnl || 0)
            return pnl >= 0 ? 'text-green-600' : 'text-red-600'
        },
        calculatePnlPercent(pos) {
            if (!pos.entry_price || !pos.current_price) return 0
            const pnlPercent = pos.side === 'long'
                ? (pos.current_price - pos.entry_price) / pos.entry_price
                : (pos.entry_price - pos.current_price) / pos.entry_price
            return pnlPercent
        },
        async loadPositions() {
            try {
                // 加载持仓
                const posRes = await axios.get('/api/live/freqtrade/positions')
                this.positions = posRes.data

                // 加载统计
                const statsRes = await axios.get('/api/live/positions/stats')
                this.stats = statsRes.data
            } catch (error) {
                console.error('加载持仓失败:', error)
            }
        },
        async closePosition(position) {
            if (!confirm(`确定要平仓 ${position.symbol} 吗？`)) return

            try {
                await axios.post('/api/live/freqtrade/close', {
                    trade_id: position.trade_id
                })
                alert('平仓请求已发送')
                this.loadPositions()
            } catch (error) {
                console.error('平仓失败:', error)
                alert('平仓失败: ' + (error.response?.data?.detail || error.message))
            }
        }
    },
    mounted() {
        this.loadPositions()

        // 监听持仓更新
        this.$root.$on('position-update', () => {
            this.loadPositions()
        })
    }
}

// 决策详情视图
const DecisionDetailView = {
    template: `
        <div>
            <div class="mb-6">
                <router-link to="/decisions" class="text-blue-600 hover:text-blue-800">
                    &larr; 返回决策列表
                </router-link>
            </div>

            <div v-if="decision" class="bg-white rounded-lg shadow p-6">
                <div class="flex justify-between items-start mb-6">
                    <div>
                        <h2 class="text-2xl font-bold mb-2">决策详情</h2>
                        <p class="text-gray-600">{{ formatTime(decision.timestamp) }}</p>
                    </div>
                    <span :class="getStatusClass(decision.status)" class="px-3 py-1 rounded-full text-sm">
                        {{ getStatusText(decision.status) }}
                    </span>
                </div>

                <div class="grid grid-cols-2 gap-6 mb-8">
                    <div>
                        <h3 class="font-semibold mb-2">基本信息</h3>
                        <table class="w-full">
                            <tr>
                                <td class="py-2 text-gray-600">交易对:</td>
                                <td class="py-2 font-medium">{{ decision.symbol }}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">操作:</td>
                                <td class="py-2">{{ formatAction(decision.action) }}</td>
                            </tr>
                            <tr>
                                <td class="py-2 text-gray-600">置信度:</td>
                                <td class="py-2">{{ decision.confidence }}%</td>
                            </tr>
                            <tr v-if="decision.position_size_usd">
                                <td class="py-2 text-gray-600">仓位大小:</td>
                                <td class="py-2">${{ formatNumber(decision.position_size_usd) }}</td>
                            </tr>
                            <tr v-if="decision.leverage">
                                <td class="py-2 text-gray-600">杠杆倍数:</td>
                                <td class="py-2">{{ decision.leverage }}x</td>
                            </tr>
                        </table>
                    </div>

                    <div>
                        <h3 class="font-semibold mb-2">风险控制</h3>
                        <table class="w-full">
                            <tr v-if="decision.stop_loss">
                                <td class="py-2 text-gray-600">止损价格:</td>
                                <td class="py-2">${{ formatNumber(decision.stop_loss) }}</td>
                            </tr>
                            <tr v-if="decision.take_profit">
                                <td class="py-2 text-gray-600">止盈价格:</td>
                                <td class="py-2">${{ formatNumber(decision.take_profit) }}</td>
                            </tr>
                            <tr v-if="decision.reason">
                                <td class="py-2 text-gray-600">决策原因:</td>
                                <td class="py-2">{{ decision.reason }}</td>
                            </tr>
                            <tr v-if="decision.agent_name">
                                <td class="py-2 text-gray-600">决策代理:</td>
                                <td class="py-2">{{ decision.agent_name }}</td>
                            </tr>
                        </table>
                    </div>
                </div>

                <!-- 层级设置 -->
                <div v-if="decision.tiers" class="mb-8">
                    <h3 class="font-semibold mb-4">层级设置</h3>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-2">层级 1</h4>
                            <p v-if="decision.tiers.tier1_target" class="text-sm">
                                目标: ${{ formatNumber(decision.tiers.tier1_target) }}
                            </p>
                            <p v-if="decision.tiers.tier1_ratio" class="text-sm">
                                平仓: {{ formatPercent(decision.tiers.tier1_ratio) }}
                            </p>
                        </div>
                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-2">层级 2</h4>
                            <p v-if="decision.tiers.tier2_target" class="text-sm">
                                目标: ${{ formatNumber(decision.tiers.tier2_target) }}
                            </p>
                            <p v-if="decision.tiers.tier2_ratio" class="text-sm">
                                平仓: {{ formatPercent(decision.tiers.tier2_ratio) }}
                            </p>
                        </div>
                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-2">层级 3</h4>
                            <p v-if="decision.tiers.tier3_target" class="text-sm">
                                目标: ${{ formatNumber(decision.tiers.tier3_target) }}
                            </p>
                            <p v-if="decision.tiers.tier3_ratio" class="text-sm">
                                平仓: {{ formatPercent(decision.tiers.tier3_ratio) }}
                            </p>
                        </div>
                    </div>
                </div>

                <!-- 执行日志 -->
                <div v-if="executionLogs.length">
                    <h3 class="font-semibold mb-4">执行日志</h3>
                    <div class="bg-gray-50 rounded p-4">
                        <pre v-for="log in executionLogs" :key="log.timestamp"
                             class="text-sm mb-2">{{ formatTime(log.timestamp) }} - {{ log.message }}</pre>
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            decision: null,
            executionLogs: []
        }
    },
    methods: {
        formatAction(action) {
            const map = {
                'enter_long': '开多',
                'enter_short': '开空',
                'close_long': '平多',
                'close_short': '平空',
                'hold': '持有'
            }
            return map[action] || action
        },
        getStatusClass(status) {
            const classes = {
                'pending': 'bg-yellow-100 text-yellow-800',
                'executed': 'bg-green-100 text-green-800',
                'failed': 'bg-red-100 text-red-800',
                'cancelled': 'bg-gray-100 text-gray-800'
            }
            return classes[status] || ''
        },
        getStatusText(status) {
            const map = {
                'pending': '待执行',
                'executed': '已执行',
                'failed': '执行失败',
                'cancelled': '已取消'
            }
            return map[status] || status
        },
        async loadDecision() {
            try {
                const id = this.$route.params.id
                const res = await axios.get(`/api/live/decisions/${id}`)
                this.decision = res.data

                // TODO: 加载执行日志
                this.executionLogs = []
            } catch (error) {
                console.error('加载决策详情失败:', error)
            }
        }
    },
    mounted() {
        this.loadDecision()
    },
    watch: {
        '$route'() {
            this.loadDecision()
        }
    }
}

// 持仓详情视图
const PositionDetailView = {
    template: `
        <div>
            <div class="mb-6">
                <router-link to="/positions" class="text-blue-600 hover:text-blue-800">
                    &larr; 返回持仓列表
                </router-link>
            </div>

            <div v-if="position" class="bg-white rounded-lg shadow p-6">
                <div class="flex justify-between items-start mb-6">
                    <div>
                        <h2 class="text-2xl font-bold mb-2">{{ position.symbol }} 持仓详情</h2>
                        <p class="text-gray-600">Trade ID: {{ position.trade_id }}</p>
                    </div>
                    <button @click="closePosition"
                            class="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700">
                        平仓
                    </button>
                </div>

                <div class="grid grid-cols-3 gap-6 mb-8">
                    <div class="col-span-2">
                        <h3 class="font-semibold mb-4">持仓信息</h3>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">方向</p>
                                <p class="text-lg font-medium">{{ position.side === 'long' ? '做多' : '做空' }}</p>
                            </div>
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">数量</p>
                                <p class="text-lg font-medium">{{ formatNumber(position.size) }}</p>
                            </div>
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">开仓价</p>
                                <p class="text-lg font-medium">${{ formatNumber(position.entry_price) }}</p>
                            </div>
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">当前价</p>
                                <p class="text-lg font-medium">${{ formatNumber(position.current_price) }}</p>
                            </div>
                        </div>
                    </div>

                    <div>
                        <h3 class="font-semibold mb-4">盈亏情况</h3>
                        <div class="space-y-4">
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">未实现盈亏</p>
                                <p :class="`text-lg font-medium ${position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`">
                                    ${{ formatNumber(position.unrealized_pnl) }}
                                </p>
                            </div>
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">已实现盈亏</p>
                                <p :class="`text-lg font-medium ${position.realized_pnl >= 0 ? 'text-green-600' : 'text-red-600'}`">
                                    ${{ formatNumber(position.realized_pnl) }}
                                </p>
                            </div>
                            <div class="border rounded p-4">
                                <p class="text-gray-600 text-sm">盈亏率</p>
                                <p :class="`text-lg font-medium ${position.profit_ratio >= 0 ? 'text-green-600' : 'text-red-600'}`">
                                    {{ formatPercent(position.profit_ratio) }}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 层级管理 -->
                <div class="mb-8">
                    <h3 class="font-semibold mb-4">层级管理</h3>
                    <div class="grid grid-cols-3 gap-4">
                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-3">层级 1</h4>
                            <div class="space-y-2">
                                <div>
                                    <label class="text-sm text-gray-600">目标价格</label>
                                    <input v-model="tiers.tier1_target" type="number" step="0.01"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                                <div>
                                    <label class="text-sm text-gray-600">平仓比例</label>
                                    <input v-model="tiers.tier1_ratio" type="number" step="0.01" min="0" max="1"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                            </div>
                        </div>

                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-3">层级 2</h4>
                            <div class="space-y-2">
                                <div>
                                    <label class="text-sm text-gray-600">目标价格</label>
                                    <input v-model="tiers.tier2_target" type="number" step="0.01"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                                <div>
                                    <label class="text-sm text-gray-600">平仓比例</label>
                                    <input v-model="tiers.tier2_ratio" type="number" step="0.01" min="0" max="1"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                            </div>
                        </div>

                        <div class="border rounded p-4">
                            <h4 class="font-medium mb-3">层级 3</h4>
                            <div class="space-y-2">
                                <div>
                                    <label class="text-sm text-gray-600">目标价格</label>
                                    <input v-model="tiers.tier3_target" type="number" step="0.01"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                                <div>
                                    <label class="text-sm text-gray-600">平仓比例</label>
                                    <input v-model="tiers.tier3_ratio" type="number" step="0.01" min="0" max="1"
                                           class="w-full px-3 py-2 border rounded">
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="mt-4">
                        <button @click="updateTiers"
                                class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700">
                            更新层级
                        </button>
                    </div>
                </div>

                <!-- 交易历史 -->
                <div v-if="trades.length">
                    <h3 class="font-semibold mb-4">交易历史</h3>
                    <table class="w-full">
                        <thead>
                            <tr class="text-left text-gray-600 border-b">
                                <th class="px-4 py-2">时间</th>
                                <th class="px-4 py-2">类型</th>
                                <th class="px-4 py-2">数量</th>
                                <th class="px-4 py-2">价格</th>
                                <th class="px-4 py-2">手续费</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="trade in trades" :key="trade.id" class="border-b">
                                <td class="px-4 py-2">{{ formatTime(trade.timestamp) }}</td>
                                <td class="px-4 py-2">{{ trade.type }}</td>
                                <td class="px-4 py-2">{{ formatNumber(trade.amount) }}</td>
                                <td class="px-4 py-2">${{ formatNumber(trade.price) }}</td>
                                <td class="px-4 py-2">${{ formatNumber(trade.fee) }}</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            position: null,
            trades: [],
            tiers: {
                tier1_target: null,
                tier1_ratio: null,
                tier2_target: null,
                tier2_ratio: null,
                tier3_target: null,
                tier3_ratio: null
            }
        }
    },
    methods: {
        async loadPosition() {
            try {
                const symbol = this.$route.params.symbol
                const res = await axios.get(`/api/live/positions/${symbol}`)
                this.position = res.data

                // 加载层级信息
                if (this.position.tier1) {
                    this.tiers.tier1_target = this.position.tier1.target
                    this.tiers.tier1_ratio = this.position.tier1.ratio
                }
                if (this.position.tier2) {
                    this.tiers.tier2_target = this.position.tier2.target
                    this.tiers.tier2_ratio = this.position.tier2.ratio
                }
                if (this.position.tier3) {
                    this.tiers.tier3_target = this.position.tier3.target
                    this.tiers.tier3_ratio = this.position.tier3.ratio
                }

                // TODO: 加载交易历史
                this.trades = []
            } catch (error) {
                console.error('加载持仓详情失败:', error)
            }
        },
        async closePosition() {
            if (!confirm(`确定要平仓 ${this.position.symbol} 吗？`)) return

            try {
                await axios.post('/api/live/freqtrade/close', {
                    trade_id: this.position.trade_id
                })
                alert('平仓请求已发送')
                this.$router.push('/positions')
            } catch (error) {
                console.error('平仓失败:', error)
                alert('平仓失败: ' + (error.response?.data?.detail || error.message))
            }
        },
        async updateTiers() {
            try {
                await axios.post('/api/live/freqtrade/tiers', {
                    trade_id: this.position.trade_id,
                    ...this.tiers
                })
                alert('层级更新成功')
            } catch (error) {
                console.error('更新层级失败:', error)
                alert('更新失败: ' + (error.response?.data?.detail || error.message))
            }
        }
    },
    mounted() {
        this.loadPosition()
    },
    watch: {
        '$route'() {
            this.loadPosition()
        }
    }
}

// 监控视图
const MonitoringView = {
    template: `
        <div>
            <h2 class="text-2xl font-bold mb-6">系统监控</h2>

            <!-- 系统状态 -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm mb-2">系统状态</h3>
                    <p :class="`text-2xl font-bold ${systemStatus.status === 'ok' ? 'text-green-600' : 'text-red-600'}`">
                        {{ systemStatus.status === 'ok' ? '正常' : '异常' }}
                    </p>
                    <p class="text-sm text-gray-500 mt-2">运行时间: {{ formatUptime(systemStatus.uptime) }}</p>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm mb-2">CPU使用率</h3>
                    <p class="text-2xl font-bold">{{ formatPercent(metrics.cpu_percent / 100) }}</p>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                        <div class="bg-blue-600 h-2 rounded-full" :style="{width: metrics.cpu_percent + '%'}"></div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm mb-2">内存使用率</h3>
                    <p class="text-2xl font-bold">{{ formatPercent(metrics.memory_percent / 100) }}</p>
                    <div class="w-full bg-gray-200 rounded-full h-2 mt-2">
                        <div class="bg-green-600 h-2 rounded-full" :style="{width: metrics.memory_percent + '%'}"></div>
                    </div>
                </div>

                <div class="bg-white rounded-lg shadow p-6">
                    <h3 class="text-gray-600 text-sm mb-2">活跃连接</h3>
                    <p class="text-2xl font-bold">{{ wsStats.total_connections }}</p>
                    <p class="text-sm text-gray-500 mt-2">WebSocket</p>
                </div>
            </div>

            <!-- 代理状态 -->
            <div class="bg-white rounded-lg shadow mb-8">
                <div class="px-6 py-4 border-b">
                    <h3 class="text-lg font-semibold">代理状态</h3>
                </div>
                <div class="p-6">
                    <table class="w-full">
                        <thead>
                            <tr class="text-left text-gray-600">
                                <th class="pb-3">代理名称</th>
                                <th class="pb-3">状态</th>
                                <th class="pb-3">最后运行</th>
                                <th class="pb-3">成功/失败</th>
                                <th class="pb-3">平均耗时</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr v-for="agent in agents" :key="agent.name" class="border-t">
                                <td class="py-3">{{ agent.name }}</td>
                                <td class="py-3">
                                    <span :class="getAgentStatusClass(agent.status)">
                                        {{ agent.status === 'running' ? '运行中' :
                                           agent.status === 'idle' ? '空闲' :
                                           agent.status === 'error' ? '错误' : agent.status }}
                                    </span>
                                </td>
                                <td class="py-3">{{ formatTime(agent.last_run) }}</td>
                                <td class="py-3">{{ agent.success_count }} / {{ agent.error_count }}</td>
                                <td class="py-3">{{ agent.avg_duration_ms }}ms</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- 最近日志 -->
            <div class="bg-white rounded-lg shadow">
                <div class="px-6 py-4 border-b flex justify-between items-center">
                    <h3 class="text-lg font-semibold">最近日志</h3>
                    <div class="flex space-x-4">
                        <select v-model="logLevel" @change="loadLogs"
                                class="px-3 py-1 border rounded text-sm">
                            <option value="">所有级别</option>
                            <option value="debug">Debug</option>
                            <option value="info">Info</option>
                            <option value="warning">Warning</option>
                            <option value="error">Error</option>
                        </select>
                        <button @click="loadLogs" class="px-3 py-1 bg-blue-600 text-white rounded text-sm">
                            刷新
                        </button>
                    </div>
                </div>
                <div class="p-6">
                    <div v-for="log in logs" :key="log.timestamp"
                         :class="`mb-2 p-2 rounded ${getLogClass(log.level)}`">
                        <span class="text-sm text-gray-500">{{ formatTime(log.timestamp) }}</span>
                        <span class="ml-2 font-medium">{{ log.level.toUpperCase() }}</span>
                        <span v-if="log.module" class="ml-2 text-gray-600">[{{ log.module }}]</span>
                        <span class="ml-2">{{ log.message }}</span>
                    </div>
                </div>
            </div>
        </div>
    `,
    data() {
        return {
            systemStatus: {},
            metrics: {},
            wsStats: {},
            agents: [],
            logs: [],
            logLevel: ''
        }
    },
    methods: {
        getAgentStatusClass(status) {
            const classes = {
                'running': 'text-green-600',
                'idle': 'text-gray-600',
                'error': 'text-red-600'
            }
            return classes[status] || ''
        },
        getLogClass(level) {
            const classes = {
                'debug': 'bg-gray-50',
                'info': 'bg-blue-50',
                'warning': 'bg-yellow-50',
                'error': 'bg-red-50'
            }
            return classes[level] || 'bg-gray-50'
        },
        formatUptime(seconds) {
            const days = Math.floor(seconds / 86400)
            const hours = Math.floor((seconds % 86400) / 3600)
            const minutes = Math.floor((seconds % 3600) / 60)

            if (days > 0) {
                return `${days}天 ${hours}小时`
            } else if (hours > 0) {
                return `${hours}小时 ${minutes}分钟`
            } else {
                return `${minutes}分钟`
            }
        },
        async loadSystemStatus() {
            try {
                const res = await axios.get('/api/live/status')
                this.systemStatus = res.data
            } catch (error) {
                console.error('加载系统状态失败:', error)
            }
        },
        async loadMetrics() {
            try {
                const res = await axios.get('/api/live/metrics')
                this.metrics = res.data
            } catch (error) {
                console.error('加载性能指标失败:', error)
            }
        },
        async loadWSStats() {
            try {
                const res = await axios.get('/api/live/ws/status')
                this.wsStats = res.data.websocket
            } catch (error) {
                console.error('加载WebSocket状态失败:', error)
            }
        },
        async loadAgents() {
            try {
                const res = await axios.get('/api/live/agents')
                this.agents = res.data
            } catch (error) {
                console.error('加载代理状态失败:', error)
            }
        },
        async loadLogs() {
            try {
                const params = {
                    limit: 50,
                    level: this.logLevel
                }
                const res = await axios.get('/api/live/logs', { params })
                this.logs = res.data.logs
            } catch (error) {
                console.error('加载日志失败:', error)
            }
        },
        async loadData() {
            this.loadSystemStatus()
            this.loadMetrics()
            this.loadWSStats()
            this.loadAgents()
            this.loadLogs()
        }
    },
    mounted() {
        this.loadData()

        // 定期刷新
        setInterval(() => {
            this.loadSystemStatus()
            this.loadMetrics()
            this.loadWSStats()
        }, 10000)
    }
}
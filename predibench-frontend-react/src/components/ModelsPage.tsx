import * as Select from '@radix-ui/react-select'
import { ChevronDown, ChevronLeft, ChevronRight, X } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import type { LeaderboardEntry, ModelInvestmentDecision, ModelPerformance } from '../api'
import { apiService } from '../api'
import { getChartColor } from './ui/chart-colors'
import { InfoTooltip } from './ui/info-tooltip'
import { ProfitDisplay } from './ui/profit-display'
import { VisxLineChart } from './ui/visx-line-chart'

interface ModelsPageProps {
  leaderboard: LeaderboardEntry[]
}


export function ModelsPage({ leaderboard }: ModelsPageProps) {
  const location = useLocation()
  const navigate = useNavigate()
  const [selectedModel, setSelectedModel] = useState<string>(leaderboard[0]?.id || '')
  const [modelDecisions, setModelDecisions] = useState<ModelInvestmentDecision[]>([])
  // const [loading, setLoading] = useState(false)
  const [selectedDate, setSelectedDate] = useState<string | null>(null)
  const [showEventPopup, setShowEventPopup] = useState(false)
  const [selectedEvent, setSelectedEvent] = useState<{
    eventDecision: any;
    decisionDate: string;
  } | null>(null)
  const [calendarDate, setCalendarDate] = useState(new Date())
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance | null>(null)

  // Popup price data for selected event
  const [popupPrices, setPopupPrices] = useState<Record<string, { date: string; price: number }[]>>({})
  const [popupMarketNames, setPopupMarketNames] = useState<Record<string, string>>({})

  const formatLongDate = (dateStr: string) => {
    if (!dateStr) return ''
    const [y, m, d] = dateStr.split('-').map(Number)
    const dt = new Date(y, (m || 1) - 1, d || 1)
    return dt.toLocaleDateString(undefined, {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    })
  }

  const selectedModelData = leaderboard.find(m => m.id === selectedModel)

  useEffect(() => {
    const urlParams = new URLSearchParams(location.search)
    const selectedFromUrl = urlParams.get('selected')

    if (selectedFromUrl && leaderboard.find(m => m.id === selectedFromUrl)) {
      setSelectedModel(selectedFromUrl)
    } else if (!selectedModel && leaderboard.length > 0) {
      setSelectedModel(leaderboard[0].id)
    }
  }, [leaderboard, selectedModel, location.search])

  // Map model_name -> model_id using performance endpoint once
  const [nameToIdMap, setNameToIdMap] = useState<Record<string, string>>({})
  useEffect(() => {
    let cancelled = false
    apiService.getPerformance('day')
      .then(perfs => {
        if (cancelled) return
        const map: Record<string, string> = {}
        perfs.forEach(p => {
          map[p.model_name] = p.model_id
          map[p.model_id] = p.model_id // allow id passthrough
        })
        setNameToIdMap(map)
      })
      .catch(console.error)
    return () => { cancelled = true }
  }, [])

  // Resolve the backend model_id to query
  const resolvedModelId = useMemo(() => nameToIdMap[selectedModel] || selectedModel, [nameToIdMap, selectedModel])
  const mapReady = useMemo(() => Object.keys(nameToIdMap).length > 0, [nameToIdMap])

  useEffect(() => {
    if (resolvedModelId && mapReady) {
      apiService.getModelResultsById(resolvedModelId)
        .then(setModelDecisions)
        .catch(console.error)
    }
  }, [resolvedModelId, mapReady])

  // Auto-select the latest decision date when modelDecisions are loaded
  useEffect(() => {
    if (modelDecisions.length > 0 && !selectedDate) {
      const sortedDates = modelDecisions
        .map(decision => decision.target_date)
        .sort((a, b) => b.localeCompare(a)) // Sort descending to get latest first

      if (sortedDates.length > 0) {
        setSelectedDate(sortedDates[0])
      }
    }
  }, [modelDecisions, selectedDate])

  // Always show calendar at current month - no override



  const handleModelSelect = (modelId: string) => {
    setSelectedModel(modelId)
    navigate(`/models?selected=${modelId}`, { replace: true })
  }

  const handleEventClick = (eventDecision: any, decisionDate: string) => {
    setSelectedEvent({ eventDecision, decisionDate })
    setShowEventPopup(true)
  }

  // Load prices for selected event in popup and calculate realized returns
  const [realizedReturns, setRealizedReturns] = useState<Record<string, number>>({})
  const [totalEventPnL, setTotalEventPnL] = useState<number>(0)
  const [positionEndDate, setPositionEndDate] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const loadPricesAndReturns = async () => {
      if (!selectedEvent) return
      try {
        const event = await apiService.getEventDetails(selectedEvent.eventDecision.event_id)
        const marketIds = new Set(
          selectedEvent.eventDecision.market_investment_decisions.map((md: any) => md.market_id)
        )

        // Find when this position ended (next decision date for same event)
        const eventDecisions = modelDecisions
          .filter(d => d.event_investment_decisions.some(ed => ed.event_id === selectedEvent.eventDecision.event_id))
          .sort((a, b) => a.target_date.localeCompare(b.target_date))

        const currentDecisionIndex = eventDecisions.findIndex(d => d.target_date === selectedEvent.decisionDate)
        const nextDecision = currentDecisionIndex >= 0 && currentDecisionIndex < eventDecisions.length - 1
          ? eventDecisions[currentDecisionIndex + 1]
          : null
        const endDate = nextDecision?.target_date || null

        const prices: Record<string, { date: string; price: number }[]> = {}
        const names: Record<string, string> = {}
        const returns: Record<string, number> = {}
        let totalPnL = 0

        event.markets.forEach(m => {
          if (marketIds.has(m.id)) {
            names[m.id] = m.question
            if (m.prices) {
              // All prices from decision date onwards
              prices[m.id] = m.prices
                .filter(p => p.date >= selectedEvent.decisionDate)
                .map(p => ({ date: p.date, price: p.value }))

              // Calculate realized returns if we have an end date
              if (endDate && m.prices.length > 0) {
                const startPrice = m.prices.find(p => p.date >= selectedEvent.decisionDate)?.value || 0
                const endPrice = m.prices
                  .filter(p => p.date <= endDate)
                  .sort((a, b) => b.date.localeCompare(a.date))[0]?.value || startPrice

                // Find the bet amount for this market
                const marketDecision = selectedEvent.eventDecision.market_investment_decisions
                  .find((md: any) => md.market_id === m.id)

                if (marketDecision) {
                  const betAmount = marketDecision.model_decision.bet
                  const priceChange = endPrice - startPrice
                  const realizedReturn = betAmount * priceChange
                  returns[m.id] = realizedReturn
                  totalPnL += realizedReturn
                }
              }
            }
          }
        })

        if (!cancelled) {
          setPopupMarketNames(names)
          setPopupPrices(prices)
          setRealizedReturns(returns)
          setTotalEventPnL(totalPnL)
          setPositionEndDate(endDate)
        }
      } catch (e) {
        console.error('Failed to load event prices and returns', e)
      }
    }
    loadPricesAndReturns()
    return () => { cancelled = true }
  }, [selectedEvent, modelDecisions])

  const navigateCalendar = (direction: 'prev' | 'next') => {
    const newDate = new Date(calendarDate)
    if (direction === 'prev') {
      newDate.setMonth(newDate.getMonth() - 1)
    } else {
      newDate.setMonth(newDate.getMonth() + 1)
    }
    setCalendarDate(newDate)
  }

  const generateCalendarDays = () => {
    // Format a Date to YYYY-MM-DD using LOCAL time to avoid timezone shifts
    const formatLocalYMD = (d: Date) => {
      const y = d.getFullYear()
      const m = String(d.getMonth() + 1).padStart(2, '0')
      const day = String(d.getDate()).padStart(2, '0')
      return `${y}-${m}-${day}`
    }

    const year = calendarDate.getFullYear()
    const month = calendarDate.getMonth()

    // First day of the month and last day of the month
    const firstDay = new Date(year, month, 1)
    const lastDay = new Date(year, month + 1, 0)

    // Start from the first Sunday of the week containing the first day
    const startDate = new Date(firstDay)
    startDate.setDate(startDate.getDate() - startDate.getDay())

    // End at the last Saturday of the week containing the last day
    const endDate = new Date(lastDay)
    endDate.setDate(endDate.getDate() + (6 - endDate.getDay()))

    const days = []
    const current = new Date(startDate)

    while (current <= endDate) {
      const dateStr = formatLocalYMD(current)
      const hasDecision = modelDecisions.some(d => d.target_date === dateStr)
      const decisionCount = hasDecision ?
        modelDecisions.find(d => d.target_date === dateStr)?.event_investment_decisions.length || 0 : 0

      const isCurrentMonth = current.getMonth() === month
      const isToday = formatLocalYMD(current) === formatLocalYMD(new Date())

      days.push({
        date: new Date(current),
        dateStr,
        hasDecision,
        decisionCount,
        isSelected: selectedDate === dateStr,
        isCurrentMonth,
        isToday
      })
      current.setDate(current.getDate() + 1)
    }

    return days
  }

  // Fetch model performance for event-level PnL series
  useEffect(() => {
    let cancelled = false
    if (resolvedModelId && mapReady) {
      apiService.getPerformanceByModel(resolvedModelId, 'day')
        .then(perf => { if (!cancelled) setModelPerformance(perf) })
        .catch(console.error)
    }
    return () => { cancelled = true }
  }, [resolvedModelId, mapReady])

  // Removed event-based series metadata; cumulative series is used instead

  // Cumulative Profit series since the beginning for the selected model
  const cumulativeSeries = useMemo(() => {
    if (!modelPerformance) return [] as { dataKey: string; data: { x: string; y: number }[]; stroke: string; name?: string }[]
    const data = (modelPerformance.cummulative_pnl || []).map(pt => ({ x: pt.date, y: pt.value }))
    return [
      {
        dataKey: `model_${modelPerformance.model_id}_cum`,
        data,
        stroke: getChartColor(0),
        name: 'Cumulative Profit'
      }
    ]
  }, [modelPerformance])

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header with title and model selection */}
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-2xl font-bold">Model performance</h1>

        <Select.Root value={selectedModel} onValueChange={handleModelSelect}>
          <Select.Trigger className="inline-flex items-center justify-between gap-2 rounded-lg border border-border bg-background px-4 py-2 text-sm font-medium text-foreground hover:bg-accent hover:text-accent-foreground focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 min-w-[300px]">
            <Select.Value placeholder="Select a model">
              {selectedModelData?.model || 'Select a model'}
            </Select.Value>
            <Select.Icon>
              <ChevronDown size={16} />
            </Select.Icon>
          </Select.Trigger>

          <Select.Portal>
            <Select.Content
              className="overflow-hidden rounded-lg border border-border bg-popover shadow-lg z-50"
              position="popper"
              side="bottom"
              align="start"
              sideOffset={4}
            >
              <Select.Viewport className="p-1 max-h-[70vh] overflow-y-auto">
                {leaderboard.map((model, index) => (
                  <Select.Item
                    key={model.id}
                    value={model.id}
                    className="relative flex cursor-pointer items-center rounded-md px-3 py-2 text-sm text-popover-foreground hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none data-[state=checked]:bg-accent data-[state=checked]:text-accent-foreground"
                  >
                    <Select.ItemText>
                      <div className="flex items-center space-x-3">
                        <div className={`flex items-center justify-center w-8 h-8 rounded-full text-xs font-bold ${index === 0 ? 'bg-gradient-to-br from-yellow-100 to-yellow-50 text-yellow-800 dark:from-yellow-900/20 dark:to-yellow-800/20 dark:text-yellow-300' :
                          index === 1 ? 'bg-gradient-to-br from-slate-100 to-slate-50 text-slate-800 dark:from-slate-900/20 dark:to-slate-800/20 dark:text-slate-300' :
                            index === 2 ? 'bg-gradient-to-br from-amber-100 to-amber-50 text-amber-800 dark:from-amber-900/20 dark:to-amber-800/20 dark:text-amber-300' :
                              'bg-gradient-to-br from-gray-100 to-gray-50 text-gray-800 dark:from-gray-900/20 dark:to-gray-800/20 dark:text-gray-300'
                          }`}>
                          {index + 1}
                        </div>
                        <div>
                          <div className="font-medium">{model.model}</div>
                          <div className="text-xs text-muted-foreground mt-1">
                            Profit: ${model.final_cumulative_pnl.toFixed(1)} | Brier score: {((1 - model.avg_brier_score) * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </Select.ItemText>
                  </Select.Item>
                ))}
              </Select.Viewport>
            </Select.Content>
          </Select.Portal>
        </Select.Root>
      </div>

      {/* Model Card Display */}
      {selectedModelData && (
        <div className="space-y-8">
          <div className="md:p-6 md:bg-card md:rounded-xl md:border md:border-border">
            {/* Model Info - Always visible */}
            <div className="mb-8 border-b border-border pb-6">
              <h2 className="text-xl font-semibold mb-4">{selectedModelData.model}</h2>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <div className="flex items-center text-muted-foreground">
                    Final Profit:
                    <InfoTooltip content="This is the PnL (Profit and Loss), or cumulative profit from all trades made by the model" />
                  </div>
                  <div className="font-medium">{selectedModelData.final_cumulative_pnl.toFixed(1)}</div>
                </div>
                <div>
                  <div className="flex items-center text-muted-foreground">
                    Brier score:
                    <InfoTooltip content="A measure of prediction accuracy. Lower values indicate better calibration - how well the model's confidence matches actual outcomes (0 = perfect, 1 = worst)" />
                  </div>
                  <div className="font-medium">{((1 - selectedModelData.avg_brier_score) * 100).toFixed(1)}%</div>
                </div >
                <div>
                  <span className="text-muted-foreground">Bets taken:</span>
                  <div className="font-medium">{selectedModelData.trades}</div>
                </div>
              </div >
            </div >

            {/* Cumulative Profit Chart */}
            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4 flex items-center">
                Cumulative Profit
                <InfoTooltip content="Cumulative profit for this model since the beginning." />
              </h3>
              <div className="h-[500px]">
                {cumulativeSeries.length === 0 ? (
                  <div className="h-full bg-muted/20 rounded-lg flex items-center justify-center">
                    <div className="text-sm text-muted-foreground">No event PnL data available.</div>
                  </div>
                ) : (
                  <VisxLineChart
                    height={500}
                    margin={{ left: 60, top: 35, bottom: 38, right: 27 }}
                    series={cumulativeSeries}
                  />
                )}
              </div>
            </div>

            {/* Decisions through time */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold">Decisions through time</h3>
              </div>
              {modelDecisions.length > 0 && (
                <div className="mb-6 p-4 bg-card rounded-lg border max-w-xl mx-auto">
                  {/* Calendar Navigation */}
                  <div className="flex items-center justify-between mb-4">
                    <button
                      onClick={() => navigateCalendar('prev')}
                      className="p-2 rounded-lg hover:bg-accent transition-colors"
                      title="Previous month"
                    >
                      <ChevronLeft size={16} />
                    </button>

                    <h4 className="font-medium text-lg">
                      {calendarDate.toLocaleDateString('en-US', { month: 'long', year: 'numeric' })}
                    </h4>

                    {(() => {
                      const today = new Date()
                      const isAtCurrentMonth =
                        calendarDate.getFullYear() === today.getFullYear() &&
                        calendarDate.getMonth() === today.getMonth()
                      return (
                        <button
                          onClick={() => !isAtCurrentMonth && navigateCalendar('next')}
                          className={`p-2 rounded-lg transition-colors ${isAtCurrentMonth ? 'opacity-50 cursor-not-allowed' : 'hover:bg-accent'
                            }`}
                          title="Next month"
                          disabled={isAtCurrentMonth}
                        >
                          <ChevronRight size={16} />
                        </button>
                      )
                    })()}
                  </div>

                  {/* Calendar Grid */}
                  <div className="grid grid-cols-7 gap-1 text-center">
                    {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map(day => (
                      <div key={day} className="p-2 text-xs font-medium text-muted-foreground">
                        {day}
                      </div>
                    ))}
                    {generateCalendarDays().map((day, index) => (
                      <div key={index} className="relative">
                        <button
                          onClick={() => day.hasDecision && setSelectedDate(day.dateStr)}
                          className={`w-full p-2 text-xs rounded transition-colors relative ${!day.isCurrentMonth
                            ? 'text-muted-foreground/30 cursor-default'
                            : day.hasDecision
                              ? day.isSelected
                                ? 'bg-primary text-primary-foreground'
                                : 'bg-accent hover:bg-accent/80 cursor-pointer'
                              : 'text-muted-foreground cursor-default'
                            } ${day.isToday ? 'ring-2 ring-white' : ''}`}
                          disabled={!day.hasDecision || !day.isCurrentMonth}
                        >
                          {day.date.getDate()}
                          {day.hasDecision && day.isCurrentMonth && (
                            <div className="absolute -top-1 -right-1 w-4 h-4 bg-primary text-primary-foreground text-xs rounded-full flex items-center justify-center">
                              {day.decisionCount}
                            </div>
                          )}
                        </button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="bg-muted/10 rounded-lg p-6">
                {modelDecisions.length > 0 && selectedDate && (
                  <div className="p-4 bg-card rounded-lg border max-w-2xl mx-auto">
                    <h4 className="font-medium mb-3">Decisions for {selectedDate}:</h4>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {modelDecisions
                        .find(d => d.target_date === selectedDate)
                        ?.event_investment_decisions.map((eventDecision, index) => (
                          <button
                            key={index}
                            onClick={() => handleEventClick(eventDecision, selectedDate)}
                            className="w-full p-3 bg-muted/20 rounded hover:bg-muted/30 transition-colors text-left h-full"
                          >
                            <div className="font-medium text-sm line-clamp-2">{eventDecision.event_title}</div>
                            <div className="text-xs text-muted-foreground mt-1">
                              {eventDecision.market_investment_decisions.length} market decisions • Click to view details
                            </div>
                          </button>
                        ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div >
        </div >
      )
      }

      {/* Event Details Popup */}
      {showEventPopup && selectedEvent && (
        <div
          className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
          onClick={() => setShowEventPopup(false)}
        >
          <div
            className="bg-card rounded-xl border border-border max-w-4xl w-full max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="sticky top-0 bg-card border-b border-border p-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold">
                {selectedEvent.eventDecision.event_title}
              </h3>
              <button
                onClick={() => setShowEventPopup(false)}
                className="p-1 rounded-lg hover:bg-accent"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-6">
              <div className="mb-6">
                <div className="text-sm text-muted-foreground mb-4">
                  Position taken on {formatLongDate(selectedEvent.decisionDate)}{positionEndDate && `, ended on ${formatLongDate(positionEndDate)}`}
                </div>

                <div className="overflow-x-auto bg-muted/10 rounded-lg p-4">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-muted-foreground">
                        <th className="text-left py-2 px-3">Market</th>
                        <th className="text-right py-2 px-3">Bet ($)</th>
                        <th className="text-right py-2 px-3">Odds</th>
                        <th className="text-right py-2 px-3">Confidence</th>
                        {positionEndDate && <th className="text-right py-2 px-3">Realized Returns</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {selectedEvent.eventDecision.market_investment_decisions.map((md: any, idx: number) => (
                        <tr key={idx} className="border-b border-border/50">
                          <td className="py-2 px-3">{md.market_question || `Market ${md.market_id}`}</td>
                          <td className="py-2 px-3 text-right">{md.model_decision.bet < 0 ? `-$${Math.abs(md.model_decision.bet).toFixed(2)}` : `$${md.model_decision.bet.toFixed(2)}`}</td>
                          <td className="py-2 px-3 text-right">{(md.model_decision.odds * 100).toFixed(1)}%</td>
                          <td className="py-2 px-3 text-right">{md.model_decision.confidence}/10</td>
                          {positionEndDate && (
                            <td className="py-2 px-3 text-right">
                              {realizedReturns[md.market_id] !== undefined ? (
                                <ProfitDisplay
                                  value={realizedReturns[md.market_id]}
                                />
                              ) : (
                                <span className="text-muted-foreground">N/A</span>
                              )}
                            </td>
                          )}
                        </tr>
                      ))}

                      {/* Unallocated capital row */}
                      <tr className="border-b border-border/50 bg-muted/10">
                        <td className="py-2 px-3 italic text-muted-foreground">Unallocated capital</td>
                        <td className="py-2 px-3 text-right">${selectedEvent.eventDecision.unallocated_capital.toFixed(2)}</td>
                        <td className="py-2 px-3"></td>
                        <td className="py-2 px-3"></td>
                        {positionEndDate && <td className="py-2 px-3"></td>}
                      </tr>

                      {/* Overall returns row */}
                      {positionEndDate && (
                        <tr className="border-t-2 border-border bg-muted/20 font-medium">
                          <td className="py-3 px-3" colSpan={4}>Overall returns</td>
                          <td className="py-3 px-3 text-right">
                            <ProfitDisplay
                              value={totalEventPnL}
                              formatValue={(v) => {
                                if (Math.abs(v) < 0.005) return '$0.00'
                                return `${v > 0 ? '+' : ''}$${v.toFixed(2)}`
                              }}
                              className="font-bold"
                            />
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>

                {/* Consolidated rationales */}
                <div className="mt-3">
                  <h4 className="font-medium mb-2">Decision rationale</h4>
                  <div className="space-y-3">
                    {selectedEvent.eventDecision.market_investment_decisions.map((md: any, idx: number) => (
                      <div key={idx} className="bg-muted/20 p-3 rounded-lg">
                        <div className="text-xs text-muted-foreground mb-2">
                          {md.market_question || `Market ${md.market_id}`}
                        </div>
                        <div className="text-sm text-foreground">
                          ▸ {md.model_decision.rationale}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Market price evolution since decision */}
              <div>
                <h4 className="font-medium mb-4">Market Prices Since Decision</h4>
                <div className="h-80">
                  {Object.keys(popupPrices).length === 0 ? (
                    <div className="h-full bg-muted/10 rounded flex items-center justify-center text-sm text-muted-foreground">
                      Loading prices or no price data.
                    </div>
                  ) : (
                    <VisxLineChart
                      height={290}
                      margin={{ left: 40, top: 20, bottom: 30, right: 20 }}
                      yDomain={[0, 1]}
                      series={Object.entries(popupPrices).map(([marketId, points], index) => ({
                        dataKey: `market_${marketId}`,
                        data: points.map(p => ({ x: p.date, y: p.price })),
                        stroke: getChartColor(index),
                        name: (popupMarketNames[marketId] || `Market ${marketId}`).substring(0, 30)
                      }))}
                    />
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div >
  )
}

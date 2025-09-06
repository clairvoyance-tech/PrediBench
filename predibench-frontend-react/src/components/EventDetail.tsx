import { ExternalLink } from 'lucide-react'
import { useEffect, useState } from 'react'
import type { Event, LeaderboardEntry } from '../api'
import { apiService } from '../api'
import { useAnalytics } from '../hooks/useAnalytics'
import { formatVolume } from '../lib/utils'
import { getChartColor } from './ui/chart-colors'
import { VisxLineChart } from './ui/visx-line-chart'

interface EventDetailProps {
  event: Event
  leaderboard: LeaderboardEntry[]
}

interface PriceData {
  date: string
  price: number
  marketId?: string
  marketName?: string
}



interface MarketInvestmentDecision {
  market_id: string
  model_name: string
  bet: number
  odds: number
  rationale: string
}

export function EventDetail({ event }: EventDetailProps) {
  const [marketPricesData, setMarketPricesData] = useState<{ [marketId: string]: PriceData[] }>({})
  const { trackEvent, trackUserAction } = useAnalytics()
  const [investmentDecisions, setInvestmentDecisions] = useState<MarketInvestmentDecision[]>([])
  const [loading, setLoading] = useState(false)
  const [latestDecisionDate, setLatestDecisionDate] = useState<string | null>(null)
  const [modelIdToName, setModelIdToName] = useState<Record<string, string>>({})

  // Function to convert URLs in text to clickable links
  const linkify = (text: string | null | undefined) => {
    if (!text) return null

    const urlRegex = /(https?:\/\/[^\s]+)/g

    return text.split(urlRegex).map((part, index) => {
      // Check if this part is a URL by testing against a fresh regex
      if (/^https?:\/\//.test(part)) {
        // Remove trailing punctuation from the URL
        const cleanUrl = part.replace(/[.,;:!?)\]]+$/, '')
        const trailingPunct = part.slice(cleanUrl.length)

        return (
          <span key={index}>
            <a
              href={cleanUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:text-blue-800 underline"
            >
              {cleanUrl}
            </a>
            {trailingPunct}
          </span>
        )
      }
      return part
    })
  }

  const loadEventDetails = async (eventId: string) => {
    setLoading(true)
    try {
      // Get market investment decisions by event
      const investmentDecisions = await apiService.getModelResultsByEvent(eventId)

      // Extract market prices from event.markets
      // Filter to last 2 months only to reduce frontend load
      const twoMonthsAgo = new Date()
      twoMonthsAgo.setMonth(twoMonthsAgo.getMonth() - 2)

      const transformedPrices: { [marketId: string]: PriceData[] } = {}
      event.markets.forEach(market => {
        if (market.prices) {
          const filteredPrices = market.prices.filter(pricePoint => {
            try {
              const priceDate = new Date(pricePoint.date)
              return priceDate >= twoMonthsAgo
            } catch {
              // If date parsing fails, include the price point to be safe
              return true
            }
          })

          transformedPrices[market.id] = filteredPrices.map(pricePoint => ({
            date: pricePoint.date,
            price: pricePoint.value,
            marketId: market.id,
            marketName: market.question
          }))
        }
      })

      // Transform investment decisions to match the component's expected format
      const transformedDecisions: MarketInvestmentDecision[] = []
      let maxDate: string | null = null
      investmentDecisions.forEach(modelResult => {
        modelResult.event_investment_decisions.forEach(eventDecision => {
          if (eventDecision.event_id === eventId) {
            // use the top-level decision target_date as the model's decision date
            if (modelResult.target_date) {
              if (!maxDate || modelResult.target_date > maxDate) maxDate = modelResult.target_date
            }
            eventDecision.market_investment_decisions.forEach(marketDecision => {
              transformedDecisions.push({
                market_id: marketDecision.market_id,
                model_name: modelResult.model_id, // keep id here; we map to pretty name for display
                bet: marketDecision.model_decision.bet,
                odds: marketDecision.model_decision.odds,
                rationale: marketDecision.model_decision.rationale
              })
            })
          }
        })
      })

      setMarketPricesData(transformedPrices)
      setInvestmentDecisions(transformedDecisions)
      setLatestDecisionDate(maxDate)
    } catch (error) {
      console.error('Error loading event details:', error)
    } finally {
      setLoading(false)
    }
  }


  useEffect(() => {
    if (event) {
      loadEventDetails(event.id)
      trackEvent('event_view', {
        event_id: event.id,
        event_title: event.title
      })
    }
  }, [event, trackEvent])

  // Build a map of model_id -> pretty model name using performance endpoint
  useEffect(() => {
    let cancelled = false
    apiService.getPerformance('day')
      .then(perfs => {
        if (cancelled) return
        const map: Record<string, string> = {}
        perfs.forEach(p => {
          map[p.model_id] = p.model_name
        })
        setModelIdToName(map)
      })
      .catch(console.error)
    return () => { cancelled = true }
  }, [])
  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <a
          href="/events"
          className="flex items-center text-muted-foreground hover:text-foreground transition-colors"
        >
          ← Back to events
        </a>
        <a
          href={`https://polymarket.com/event/${event.slug}`}
          target="_blank"
          rel="noopener noreferrer"
          onClick={() => trackUserAction('external_link_click', 'engagement', 'polymarket')}
          className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
        >
          Visit on Polymarket
          <ExternalLink className="h-4 w-4 ml-2" />
        </a>
      </div>

      <div>
        {/* Title */}
        <div className="mb-4">
          <h1 className="text-4xl font-bold mb-4">{event.title}</h1>

          {/* Status indicators and info */}
          <div className="flex items-center space-x-4">
            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
              <div className="w-2 h-2 bg-green-500 rounded-full mr-1"></div>
              {event.end_datetime && new Date(event.end_datetime) > new Date() ? 'LIVE' : 'CLOSED'}
            </span>

            <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-blue-50 text-blue-900 border border-blue-200">
              <span className="font-medium">Volume:</span>
              <span className="ml-1">{formatVolume(event.volume)}</span>
            </div>

            <div className="inline-flex items-center px-4 py-2 rounded-full text-sm font-medium bg-gray-50 text-gray-900 border border-gray-200">
              <span className="font-medium">Ends:</span>
              <span className="ml-1">{event.end_datetime ? new Date(event.end_datetime).toLocaleDateString('en-US') : 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Market Price Charts - Superposed */}
        <div>
          {loading ? (
            <div className="h-64 flex items-center justify-center">
              <div className="text-center">
                <div className="w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-2"></div>
                <div className="text-sm text-muted-foreground">Loading market data...</div>
              </div>
            </div>
          ) : (
            <div className="w-full">
              {/* Market Legend */}
              <div className="mb-4 flex flex-wrap gap-2">
                {event?.markets?.map((market, index) => (
                  <div key={market.id} className="flex items-center space-x-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{
                        backgroundColor: getChartColor(index)
                      }}
                    ></div>
                    <span className="text-sm text-muted-foreground">
                      {market.question.length > 50 ? market.question.substring(0, 47) + '...' : market.question}
                    </span>
                  </div>
                ))}
              </div>

              <div className="w-full h-96">
                <VisxLineChart
                  height={384}
                  margin={{ left: 60, top: 35, bottom: 38, right: 27 }}
                  yDomain={[0, 1]}
                  series={event?.markets?.map((market, index) => ({
                    dataKey: `market_${market.id}`,
                    data: (marketPricesData[market.id] || []).map(point => ({
                      x: point.date,
                      y: point.price
                    })),
                    stroke: getChartColor(index),
                    name: market.question.length > 30 ? market.question.substring(0, 27) + '...' : market.question
                  })) || []}
                />
              </div>
            </div>
          )}
        </div>

        {/* Event Description */}
        <div className="mt-8 mb-8">
          <div className="text-muted-foreground text-base leading-relaxed">
            {linkify(event.description)}
          </div>
        </div>

        {/* Latest Model Predictions */}
        <div className="mt-8">
          <h2 className="text-2xl font-bold mb-6">
            Latest Predictions{latestDecisionDate ? ` (${new Date(latestDecisionDate).toLocaleDateString('en-US', { day: 'numeric', month: 'long', year: 'numeric' })})` : ''}
          </h2>

          {loading ? (
            <div className="flex items-center justify-center py-8">
              <div className="text-center">
                <div className="w-6 h-6 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-2"></div>
                <div className="text-sm text-muted-foreground">Loading predictions...</div>
              </div>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border">
                    <th className="text-left py-3 px-4 text-sm font-medium text-muted-foreground"></th>
                    {/* Create column headers for each unique model */}
                    {[...new Set(investmentDecisions.map(decision => decision.model_name))].map(modelId => (
                      <th key={modelId} className="text-center py-3 px-4 text-sm font-medium text-muted-foreground">
                        <a href={`/models?selected=${encodeURIComponent(modelId)}`} className="text-foreground hover:underline">
                          {modelIdToName[modelId] || modelId}
                        </a>
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {/* Bet row */}
                  <tr className="border-b border-border bg-muted/50">
                    <td className="py-3 px-4 font-medium text-sm">Bet</td>
                    {[...new Set(investmentDecisions.map(decision => decision.model_name))].map(modelId => {
                      const decision = investmentDecisions.find(d => d.model_name === modelId)
                      return (
                        <td key={modelId} className="py-3 px-4 text-center">
                          {decision && (
                            <span className={`px-2 py-1 rounded-full text-xs font-medium ${decision.bet < 0
                              ? 'bg-green-100 text-green-800'
                              : 'bg-red-100 text-red-800'
                              }`}>
                              {decision.bet.toFixed(2)}
                            </span>
                          )}
                        </td>
                      )
                    })}
                  </tr>

                  {/* Confidence row */}
                  <tr>
                    <td className="py-3 px-4 font-medium text-sm">Confidence</td>
                    {[...new Set(investmentDecisions.map(decision => decision.model_name))].map(modelId => {
                      const decision = investmentDecisions.find(d => d.model_name === modelId)
                      return (
                        <td key={modelId} className="py-3 px-4 text-center text-sm text-muted-foreground">
                          {decision && `${(decision.odds * 100).toFixed(0)}%`}
                        </td>
                      )
                    })}
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

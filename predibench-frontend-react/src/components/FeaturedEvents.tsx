import { Search } from 'lucide-react'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import type { Event } from '../api'
import { Card, CardContent, CardHeader } from './ui/card'
import { EventCard } from './EventCard'

interface FeaturedEventsProps {
  events: Event[]
  loading?: boolean
  showTitle?: boolean
  maxEvents?: number
  showFilters?: boolean
}

export function FeaturedEvents({ 
  events, 
  loading = false, 
  showTitle = true, 
  maxEvents = 6,
  showFilters = true
}: FeaturedEventsProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [sortBy, setSortBy] = useState<'volume' | 'probability' | 'endDate'>('volume')
  const [orderBy, setOrderBy] = useState<'asc' | 'desc'>('desc')
  const [isLive, setIsLive] = useState(false)
  const [selectedTag, setSelectedTag] = useState<string>('')

  // Get unique tags from all events
  const uniqueTags = Array.from(new Set(events.flatMap(event => event.tags || []))).sort()

  // Filter and sort events
  const filteredAndSortedEvents = events
    .filter(event => {
      const matchesSearch = searchQuery === '' ||
        event.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        event.markets?.some(market =>
          market.question?.toLowerCase().includes(searchQuery.toLowerCase())
        )

      const matchesStatus = isLive ? (event.end_datetime ? new Date(event.end_datetime) > new Date() : true) : true
      
      const matchesTag = selectedTag === '' || (event.tags && event.tags.includes(selectedTag))

      // Only show events with markets that have multiple datapoints in their timeseries
      const hasMultipleDatapoints = event.markets?.some(market => 
        market.prices && market.prices.length > 1
      ) ?? false

      return matchesSearch && matchesStatus && matchesTag && hasMultipleDatapoints
    })
    .sort((a, b) => {
      let comparison = 0
      switch (sortBy) {
        case 'volume':
          comparison = (a.volume || 0) - (b.volume || 0)
          break
        case 'probability': {
          const aAvgProb = a.markets?.reduce((sum, m) => sum + (m.outcomes[0].price), 0) / (a.markets.length)
          const bAvgProb = b.markets?.reduce((sum, m) => sum + (m.outcomes[0].price), 0) / (b.markets.length)
          comparison = (aAvgProb) - (bAvgProb)
          break
        }
        case 'endDate':
          comparison = new Date(a.end_datetime || '').getTime() - new Date(b.end_datetime || '').getTime()
          break
      }
      return orderBy === 'desc' ? -comparison : comparison
    })

  return (
    <div>
      {showTitle && (
        <div className="text-center mb-8">
          <div className="flex items-center justify-center space-x-4">
            <h2 className="text-2xl font-bold">Featured Events</h2>
            <Link 
              to="/events" 
              className="text-primary hover:text-primary/80 transition-colors font-medium"
            >
              View all →
            </Link>
          </div>
        </div>
      )}
      
      {/* Search and Filters */}
      {showFilters && (
        <div className="mb-8 space-y-4">
          {/* Search Bar */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search events by title, topic, ticker, or markets..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-3 border border-border rounded-lg bg-background focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
            />
          </div>

          {/* Filters */}
          <div className="flex flex-wrap items-center gap-4">
            {/* Tag Filter */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Tag:</span>
              <select
                value={selectedTag}
                onChange={(e) => setSelectedTag(e.target.value)}
                className="px-3 py-1 border border-border rounded bg-background text-sm"
                disabled={uniqueTags.length === 0}
              >
                <option value="">All tags</option>
                {uniqueTags.length === 0 ? (
                  <option disabled>No tags available</option>
                ) : (
                  uniqueTags.map(tag => (
                    <option key={tag} value={tag}>{tag}</option>
                  ))
                )}
              </select>
            </div>
            
            {/* Sort By */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Sort by:</span>
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as 'volume' | 'probability' | 'endDate')}
                className="px-3 py-1 border border-border rounded bg-background text-sm"
              >
                <option value="volume">Volume</option>
                <option value="probability">Probability</option>
                <option value="endDate">End Date</option>
              </select>
            </div>

            {/* Order */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Order:</span>
              <select
                value={orderBy}
                onChange={(e) => setOrderBy(e.target.value as 'asc' | 'desc')}
                className="px-3 py-1 border border-border rounded bg-background text-sm"
              >
                <option value="desc">High to Low</option>
                <option value="asc">Low to High</option>
              </select>
            </div>

            {/* Live/All Toggle */}
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Status:</span>
              <button
                onClick={() => setIsLive(!isLive)}
                className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                  isLive ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-800'
                }`}
              >
                {isLive ? 'Live' : 'All'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading Spinner when initially loading */}
      {loading && events.length === 0 && (
        <div className="flex items-center justify-center py-16">
          <div className="text-center">
            <div className="w-8 h-8 border-4 border-primary/20 border-t-primary rounded-full animate-spin mx-auto mb-2"></div>
            <div className="text-sm text-muted-foreground">Loading events...</div>
          </div>
        </div>
      )}

      {/* Events Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading && events.length === 0 ? (
          Array.from({ length: maxEvents }).map((_, index) => (
            <Card key={index}>
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1 space-y-2">
                    <div className="h-4 bg-gray-200 rounded animate-pulse"></div>
                    <div className="h-3 bg-gray-200 rounded animate-pulse w-3/4"></div>
                  </div>
                  <div className="flex space-x-2 ml-2">
                    <div className="h-6 w-16 bg-gray-200 rounded-full animate-pulse"></div>
                    <div className="h-6 w-12 bg-gray-200 rounded-full animate-pulse"></div>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0">
                <div className="h-20 bg-gray-200 rounded animate-pulse"></div>
              </CardContent>
            </Card>
          ))
        ) : filteredAndSortedEvents.length === 0 ? (
          <div className="col-span-full text-center py-12">
            <p className="text-muted-foreground">No events found matching your search criteria.</p>
          </div>
        ) : (
          filteredAndSortedEvents.slice(0, maxEvents).map((event) => (
            <EventCard key={event.id} event={event} />
          ))
        )}
      </div>
    </div>
  )
}
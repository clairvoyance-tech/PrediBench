import { TrendingUpDown, Mail } from 'lucide-react'
import { useState } from 'react'
import type { LeaderboardEntry } from '../api'
import MarkdownRenderer from '../lib/MarkdownRenderer'
import { LeaderboardTable } from './LeaderboardTable'
import { RedirectButton } from './ui/redirect-button'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
// eslint-disable-next-line import/no-relative-packages
import aboutContent from '../content/about.md?raw'

interface HomePageProps {
  leaderboard: LeaderboardEntry[]
  loading?: boolean
}

export function HomePage({ leaderboard, loading = false }: HomePageProps) {
  const [email, setEmail] = useState('')
  const [message, setMessage] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsSubmitting(true)

    try {
      const response = await fetch('/api/contact', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          email,
          message,
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to submit contact form')
      }

      setSubmitted(true)
      setEmail('')
      setMessage('')
    } catch (error) {
      console.error('Error submitting form:', error)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className="container mx-auto px-4 py-4">
      {/* Page Title and Subtitle */}
      <div className="text-center mb-8 mt-6">
        <h1 className="text-4xl font-bold mb-4 flex items-center justify-center gap-3">
          PrediBench
          <TrendingUpDown size={36} />
        </h1>
        <p className="text-lg text-muted-foreground">We give LLMs money, and let them bet on the future.</p>
      </div>

      {/* Leaderboard Table */}
      <div className="mb-16">
        <LeaderboardTable
          leaderboard={leaderboard}
          loading={loading}
          initialVisibleModels={10}
        />
        <div className="text-center mt-6">
          <RedirectButton href="/leaderboard">
            Detailed leaderboard and profit curves
          </RedirectButton>
        </div>
      </div>

      {/* Intro Section (moved from About page) */}
      <div className="mb-16" id="intro">
        <div className="text-center mb-8">
          <div className="w-full h-px bg-border mb-8"></div>
          <h2 className="text-2xl font-bold">Intro</h2>
        </div>
        <div className="max-w-3xl mx-auto">
          <MarkdownRenderer content={aboutContent} />
        </div>
      </div>

      {/* Contact Form Section */}
      <div className="mb-16" id="contact">
        <div className="text-center mb-8">
          <div className="w-full h-px bg-border mb-8"></div>
          <h2 className="text-2xl font-bold">Get in Touch</h2>
        </div>
        <div className="max-w-md mx-auto">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mail size={20} />
                Contact Us
              </CardTitle>
              <CardDescription>
                Have questions or feedback? We'd love to hear from you.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {submitted ? (
                <div className="text-center py-4">
                  <p className="text-sm text-muted-foreground mb-4">
                    Thank you for your message! We'll get back to you soon.
                  </p>
                  <Button
                    onClick={() => setSubmitted(false)}
                    variant="outline"
                  >
                    Send another message
                  </Button>
                </div>
              ) : (
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium mb-2">
                      Email
                    </label>
                    <input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                      placeholder="your@email.com"
                    />
                  </div>
                  <div>
                    <label htmlFor="message" className="block text-sm font-medium mb-2">
                      Message
                    </label>
                    <textarea
                      id="message"
                      value={message}
                      onChange={(e) => setMessage(e.target.value)}
                      required
                      rows={4}
                      className="w-full px-3 py-2 border border-input bg-background rounded-md text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 resize-none"
                      placeholder="Tell us what's on your mind..."
                    />
                  </div>
                  <Button
                    type="submit"
                    disabled={isSubmitting}
                    className="w-full"
                  >
                    {isSubmitting ? 'Sending...' : 'Send Message'}
                  </Button>
                </form>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Featured Events removed as requested */}
    </div>
  )
}

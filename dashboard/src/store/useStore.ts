import { create } from 'zustand'

export type NearMissEvent = {
  id: string
  timestamp: string
  yamnet_class_id: number
  yamnet_class: string
  anomaly_score: number
  gps_lat: number
  gps_lon: number
  status: 'PENDING' | 'CONFIRMED' | 'DISMISSED'
  audio_url?: string
  frame_url?: string
}

type StoreState = {
  events: NearMissEvent[]
  activeSite: string
  addEvent: (event: NearMissEvent) => void
  setEvents: (events: NearMissEvent[]) => void
  updateEventStatus: (id: string, status: NearMissEvent['status']) => void
  setActiveSite: (siteId: string) => void
}

export const useStore = create<StoreState>((set) => ({
  events: [],
  activeSite: 'site_1',
  addEvent: (event) => set((state) => ({ events: [event, ...state.events] })),
  setEvents: (events) => set({ events }),
  updateEventStatus: (id, status) => set((state) => ({
    events: state.events.map((e) => e.id === id ? { ...e, status } : e)
  })),
  setActiveSite: (siteId) => set({ activeSite: siteId })
}))

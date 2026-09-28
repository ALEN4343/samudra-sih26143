---
title: SAMUDRA
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
pinned: true
short_description: Oil-spill detection, drift and vessel attribution
---

# SAMUDRA — oil-spill attribution engine

Smart India Hackathon — SIH26143 (NTRO). Detects an oil spill in satellite radar, traces it
back to its origin with a drift model, filters AIS ship traffic to the vessels that were
there at the right time, ranks them by simulating a discharge from each, and forecasts where
the spill goes next.

The app opens on the investigator view. The cases shown are exercises with a planted
answer, so the result can be checked; they are labelled *Simulation mode*. Model output is
decision-support evidence and does not establish legal responsibility.

Code, documentation and the trained model: <https://github.com/ALEN4343/samudra-sih26143>

---
layout: default
title: Home
---

# Welcome 2026 👋 -by SonJW


{% for post in site.posts %}
- **[{{ post.title }}]({{ post.url }})**  
  <span style="color:#666">{{ post.paper }} · {{ post.date | date: "%Y-%m-%d" }}</span>
{% endfor %}


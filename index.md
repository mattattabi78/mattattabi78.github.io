---
layout: default
title: Home
---

# Welcome 2026 👋 -by SonJW


<ul class="post-list">
{% for post in site.posts %}
  <li class="post-item">
    <a class="post-title" href="{{ post.url }}">
      {{ post.title }}
    </a>

    <div class="post-meta">
      {{ post.date | date: "%Y-%m-%d" }}
    </div>

    <div class="post-desc">
      {{ post.summary }}
    </div>
  </li>
{% endfor %}
</ul>

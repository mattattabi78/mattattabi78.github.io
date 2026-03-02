---
layout: default
title: Home
---

# Welcome 2026 👋 -by SonJW


<ul class="post-list">
{% for post in site.posts %}
  <li class="post-item">

    {% if post.thumbnail %}
      <div class="post-thumb">
        <a href="{{ post.url | relative_url }}">
          <img src="{{ post.thumbnail | relative_url }}" alt="{{ post.title }}">
        </a>
      </div>
    {% endif %}

    <div class="post-content">
      <a class="post-title" href="{{ post.url | relative_url }}">
        {{ post.title }}
      </a>

      <div class="post-meta">
        {{ post.date | date: "%Y-%m-%d" }}
      </div>

      <div class="post-desc">
        {{ post.summary | default: post.excerpt | strip_html | truncate: 120 }}
      </div>
    </div>

  </li>
{% endfor %}
</ul>

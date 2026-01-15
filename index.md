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

    {% if post.paper %}
      <div class="post-meta">
        {{ post.paper }}
      </div>
    {% endif %}
  </li>
{% endfor %}
</ul>

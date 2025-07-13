{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :undoc-members:
   :show-inheritance:

   {% block methods %}
   {% if methods %}
   .. autosummary::
      :toctree: .
   {% for item in methods %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. autosummary::
      :toctree: .
   {% for item in attributes %}
      {{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}